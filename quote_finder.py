import pandas as pd
import pickle
from bs4 import BeautifulSoup
import tokenizers
import argparse
import ollama
import json
import os
import re
import sentence_splitter

class Summarizer:

    def __init__(self, backstory_strength):
        self.model_name = 'mistral'
        self.tokenizer = tokenizers.Tokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
        self.reasonable_text_length = 32000
        self.model_token_limit = 4*1024 # anyone who tells you that Mistral can process 8k tokens is lying
        self.context_ratio = 0.5  # context is allowed to take up this percentage of the context window space before we start compressing context
        self.backstory_strength = backstory_strength
        self.ollama_options = {
            'num_ctx': self.model_token_limit,
            'temperature': 0.0,
            'num_predict': 38400,
            'top_k': 20,
            'top_p': 0.5
        }

        self.prompt_template = """
        --- Start of text section ---
        {full_text}
        --- End of text section ---

        Response format: english, extract direct quotes and intertextual references in the text followed by the entity referenced in parentheses, separate each reference or quote with /n.
        Quotes and references in the text:

        """

    def split_text(self, text, token_limit, from_the_end=False, contingency=True):
        if len(text) == 0:
            return '', ''

        # Check if the entire text happens to be shorter than the token limit
        text_length = self.measure_tokens(text)
        if text_length < token_limit:
            return text, ''

        # Avoid splitting the text in the middle of a sentence
        sentences = split(text)
        if from_the_end:
            sentences.reverse()

        # Contingency: if we're not more than twice over the token limit, we should split the text in half
        if contingency:
            if text_length < token_limit * 2:
                token_limit = text_length // 2

        # Run a binary search to find the split point that fits the maximum amount of text into the token limit
        low = 0
        high = len(sentences)
        best_split_point = 0
        while low < high:
            mid = (low + high) // 2
            first_part_test = ' '.join(sentences[:mid])
            test_length = self.measure_tokens(first_part_test)
            if test_length < token_limit:
                best_split_point = mid  # this split is valid, might be the best one so far
                low = mid + 1  # look for a split that allows more text in the first part
            else:
                high = mid  # look for a split that puts less text in the first part
        sentence_split_point = best_split_point

        if from_the_end:
            sentences.reverse()
            last_part = ' '.join(sentences[-sentence_split_point:])
            if sentence_split_point == len(sentences):
                remaining_part = ''
            else:
                remaining_part = ' '.join(sentences[:-sentence_split_point])
            return last_part, remaining_part

        first_part = ' '.join(sentences[:sentence_split_point])
        if sentence_split_point == len(sentences):
            remaining_part = ''
        else:
            remaining_part = ' '.join(sentences[sentence_split_point:])
        return first_part, remaining_part

    def measure_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens.ids)

    def condense_text(self, input_text, condensed_text):
        # check if the invariate parts of the prompt exceed our token limit
        if len(condensed_text) > 0:
            summary_prompt = self.prompt_template.format(backstory=condensed_text)
            summary_length = self.measure_tokens(summary_prompt)
        else:
            summary_prompt = ''
            summary_length = 0
        template_length = self.measure_tokens(self.prompt_template)
        if summary_length + template_length > self.model_token_limit:
            raise ValueError('This text is too long to summarize, at least with the default network. Try using one with a longer context window!')

        # try to squeeze as much input text as close as possible into the prompt
        chunk_limit = self.model_token_limit - (summary_length + template_length)
        text_chunk, remaining_chunk = self.split_text(input_text, chunk_limit)
        main_prompt = self.prompt_template.format(full_text=text_chunk)
        prompt = summary_prompt + main_prompt
        prompt_length = self.measure_tokens(prompt)

        # ask an LLM to condense the full text for us
        text_chunk_length = self.measure_tokens(text_chunk)
        overhead_ratio = (prompt_length - text_chunk_length) / self.model_token_limit
        print(f'New prompt: {text_chunk_length:d} tokens of text plus {overhead_ratio:.0%} overhead')
        print(f'Sending {prompt_length:d} tokens to the LLM...')
        output = ollama.generate(model=self.model_name, prompt=prompt, options=self.ollama_options)

        # gather the LLM output
        response = output['response']
        response_length = self.measure_tokens(response)
        processing_time = output['total_duration'] / 1000000000
        if len(text_chunk) is not None:
            remaining_text_length = self.measure_tokens(remaining_chunk)
            processing_speed = (prompt_length + response_length) / processing_time
            print(f'...finished in {processing_time:.0f} seconds ({processing_speed:.0f} t/s).')
            print(f'Quotes: {response_length:d} tokens')
            print(f'Remaining input text: {remaining_text_length:d} tokens')
        return response, remaining_chunk


    def create_context(self, round_index, section_index):
        responses = self.responses[round_index]
        context = '\n\n'.join(responses)

        # check if we take up too much room in the context window
        context_token_size = self.measure_tokens(context)
        context_token_limit = self.model_token_limit * self.context_ratio
        if context_token_size < context_token_limit:
            return context

        # weak backstory means that we forget the first part of the context once our context space is exceeded
        if self.backstory_strength == 'weak' and section_index is not None:
            text_chunk, remaining_chunk = self.split_text(context, context_token_limit, from_the_end=True, contingency=False)
            discard_ratio = len(remaining_chunk) / len(context)
            print(f'Context limit exceeded, omitting the first {discard_ratio:.0%} of context...')
            return text_chunk

        # strong backstory means that we condense the previous condensed context even further
        input_context = context
        condensed_context = ''
        input_context_length = self.measure_tokens(input_context)
        print(f'\nCondensing {input_context_length:d} tokens of context before continuing...')
        # this needs to be done every time a new text chunk comes in, so it takes more time overall
        while True:
            # process the text in chunks small enough to fit into the LLM's context window
            condensate, remaining_context = self.condense_text(input_context, condensed_context)
            condensed_context += condensate

            # check if we can wrap up this round because the entire input text has been condensed
            if len(remaining_context) == 0:
                break

            # not done? aw. prepare for the next round
            input_context = remaining_context

        print('...finished.')
        return condensed_context

    def summarize(self, input_text):
        resultados = {"indice":"", "quotes":""}
        input_text_length = self.measure_tokens(input_text)
        print(f'Trying to find quotes in a text with a length of {input_text_length:d} tokens...')

        run_index = 0
        self.responses = []

        while True:
            # run the compression stage, replacing the input text with the condensed text, until the output is short enough
            condensed_text = ''
            section_index = 0
            self.responses.append([])
            while True:
                # process the text in chunks small enough to fit into the LLM's context window
                print(f'\nRound {run_index+1:d}-{section_index+1:d}...')
                response, remaining_chunk = self.condense_text(input_text, condensed_text)

                # store the results
                self.responses[run_index].append(response)

                # check if we can wrap up this round because the entire input text has been condensed
                if len(remaining_chunk) == 0:
                    break

                # not done? aw. prepare for the next round
                section_index += 1
                if self.backstory_strength in ('weak', 'strong'):
                    condensed_text = self.create_context(run_index, section_index)
                else:
                    condensed_text = ''
                input_text = remaining_chunk

            # collect all the condensed text from this round
            quotes = self.create_context(run_index, None)
            condensation_factor = quotes.count("\n")
            print(f'...finished round {run_index+1:d}! found {condensation_factor + 1:d} quotes.')
            
            quotes = quotes.splitlines()
            resultados["indice"] = run_index
            resultados["quotes"] = quotes
            print(f'Processing complete!')
            break
            run_index += 1

        return resultados

def main():
    arg_parser = argparse.ArgumentParser(description='Finds quotes in a dataset of news articles')
    arg_parser.add_argument('book_folderpath', help='The path to the folder that contains the EXCEL')
    arg_parser.add_argument('--backstory_strength', default='strong', help='How strongly the the previous context should be tracked (none/weak/strong)')
    args = arg_parser.parse_args()
    book_folderpath = args.book_folderpath
    backstory_strength = args.backstory_strength
    book_summarizer = Summarizer(backstory_strength)
    excel = pd.read_excel(excel, sheet_name="Texto")
    for text in excel["text"]:
      condensed_text = book_summarizer.summarize(text)
      print(condensed_text)

if __name__ == '__main__':
    main()
