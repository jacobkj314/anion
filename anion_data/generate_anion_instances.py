import pandas, re, json

#Here are some helper functions for formatting 
def infinitive(phrase:str): #some inference types require infinitive phrases to fit the prompt templates, but this was not consistently followed
    return phrase if phrase[:2] == 'to' else 'to '+phrase
def clean(phrase):
    return re.sub(r'(?<=person) (?=[xy])', r'', # standardize personx/persony spelling
           re.sub(r'\.(?=[,?])', r'', phrase    # remove extra punctuation
           ).lower().capitalize())              # standardize capitalization
def format_instance(prompt, answer):
    return json.dumps({"input":[prompt], "answer":[answer]})+"\n"

#plausibility classifier prompt templates (from https://aclanthology.org/2021.naacl-main.346.pdf page 4392) adapted into questions
_formats = {
    "xIntent" : (lambda event,inference : clean(f'Is it plausible that {event} because PersonX wanted {infinitive(inference)}?')),
    "xNeed"   : (lambda event,inference : clean(f'Is it plausible that, before {event}, PersonX needed {infinitive(inference)}?')),
    "xAttr"   : (lambda event,inference : clean(f'Is it plausible that, if {event}, PersonX is seen as {inference}?')),
    "xWant"   : (lambda event,inference : clean(f'Is it plausible that, because {event}, PersonX wants {infinitive(inference)}?')),
    "oWant"   : (lambda event,inference : clean(f'Is it plausible that, because {event}, others want {infinitive(inference)}?')),
    "xEffect" : (lambda event,inference : clean(f'Is it plausible that, because {event}, PersonX then {inference}?')),
    "oEffect" : (lambda event,inference : clean(f'Is it plausible that, because {event}, others then {inference}?')),
    "xReact"  : (lambda event,inference : clean(f'Is it plausible that, because {event}, PersonX feels {inference}?')),
    "oReact"  : (lambda event,inference : clean(f'Is it plausible that, because {event}, others feel {inference}?'))
}


_splits = ['trn', 'dev', 'tst']

_positive_types = ['original']
_negative_types = ['logical_neg', 'semi_logical_neg', 'commonsense_contradict']
_negation_types = _positive_types + _negative_types

#this creates the big dictionary
dataset = {negation_type:dict() for negation_type in _negation_types}
for split in _splits:
    for negation_type in _negation_types:
        filepath = f'{negation_type}/{negation_type}_{split}.csv'
        table = pandas.read_csv(filepath)
        for i,row in table.iterrows():
            event = row['event']
            if '_' not in event: # Filter out  instances with blanks
                original = (row['original'] 
                            if 'original' in row else 
                            row['event'])
                event_dict = dataset[negation_type][original] = dict()
                event_dict['event'] = event
                for inference_type, format in _formats.items():
                    inferences = eval(row[inference_type]) if isinstance(row[inference_type], str) else [] # slightly janky eval to get list from csv string :shrug:
                    event_dict[inference_type] = [inference for inference in inferences if inference != "none" and '_' not in inference]

#use the big dictionary to print the samples
for split in _splits:
    for negation_type in _negation_types:
        filepath = f'{negation_type}/{negation_type}_{split}.csv'
        table = pandas.read_csv(filepath)
        with open(f'{negation_type}/{negation_type}_{split}_formatted.json', "w") as writer:
            for i,row in table.iterrows():
                event = row['event']
                if '_' not in event: # Filter out  instances with blanks
                    #POSITIVE INSTANCES
                    for inference_type, format_prompt in _formats.items():
                        inferences = eval(row[inference_type]) if isinstance(row[inference_type], str) else [] # slightly janky eval to get list from csv string :shrug:
                        inferences = [inference for inference in inferences if inference != "none" and '_' not in inference]
                        for inference in inferences:
                            prompt = format_prompt(event, inference)
                            instance = format_instance(prompt, 'yes')
                            try:
                                writer.write(instance)
                            except UnicodeEncodeError:
                                pass
                    #NEGATIVE INSTANCES
                    original = (row['original'] 
                            if 'original' in row else 
                            row['event'])
                    for opposing_negation_type in (_negative_types if negation_type in _positive_types else _positive_types):
                        if original in dataset[opposing_negation_type]:
                            for inference_type, format_prompt in _formats.items():
                                opposing_inferences = dataset[opposing_negation_type][original][inference_type]
                                for opposing_inference in opposing_inferences:
                                    if opposing_inference not in dataset[negation_type][original][inference_type]:
                                        prompt = format_prompt(event, opposing_inference)
                                        instance = format_instance(prompt, 'no')
                                        try:
                                            writer.write(instance)
                                        except UnicodeEncodeError: #
                                            pass

