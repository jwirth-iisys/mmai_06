import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "Qwen/Qwen3-ASR-1.7B-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")


import torchaudio




dataset_paths = [#("/training-1/voxpopuli/dataset_with_spaces_voxpopuli_en/test", "voxpopuli"),
                 #("/training-1/asr/kaldi_data/cv19/en/test", "commonvoice"),
                 #("/training-1/asr/kaldi_data/singleword/en/test", "commonvoice_singlewords"),
                 ("/training-1/asr/kaldi_data/mls/en/test", "librispeech")

                 
                ]# other datasets
# Mozilla Commonvoice: /training-1/asr/kaldi_data/cv19/en/test /de/ /fr/ /es/
# Mozilla Commonvoice Single Words: /training-1/asr/kaldi_data/singleword/en/test /de/ /fr/ /es/ 
# Multilingual Librispeech: /training-1/asr/kaldi_data/mls/en/test /de/ /fr/ /es/
# Voxpopuli: /training-1/voxpopuli/dataset_with_spaces_voxpopuli_en/test *_de *_fr *_es
datasets = {}
for dataset_path, dataset_title in dataset_paths:
    dataset = []
    wav_scp_dict = {}
    with open(f"{dataset_path}/wav.scp", encoding="UTF-8") as rf:
        for line in rf:
            id, wav_path = line.strip().split("\t")
            wav_scp_dict[id] = wav_path
    
    text_dict = {}
    with open(f"{dataset_path}/text", encoding="UTF-8") as rf:
        for line in rf:
            id, text = line.strip().split("\t")
            text_dict[id] = text
    
    for id, wav_path in wav_scp_dict.items():
        text = text_dict[id]
        dataset.append({"audio" : wav_path, "text" : text})
    datasets[dataset_title] = dataset

#helper function: load audio and make sure it is at 16kHz
def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    waveform = torchaudio.functional.resample(
        waveform, sample_rate, 16000
    )

    waveform = waveform.mean(dim=0)
    return waveform
'''
first_entry = dataset[0]
print(first_entry)
print("-"*10)
print(first_entry["text"])
audio = load_audio(first_entry["audio"])
display(Audio(audio, rate=16000))
'''
def transcribe(audio_path, prompt=None, language=None):
    kwargs = {"audio": audio_path}
    if prompt:
        kwargs["prompt"] = prompt
    if language:
        kwargs["language"] = language
    inputs = processor.apply_transcription_request(**kwargs).to(model.device, model.dtype)

    output_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    return processor.decode(generated_ids, return_format="transcription_only")[0].strip()



import jiwer
import re
import string

def normalize(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    text = re.sub(r" +", " ", text).strip()
    return text



from tqdm.auto import tqdm
import pandas as pd
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_trf")
context_list = [False, True]

for c in context_list:
    for dataset_title, dataset_entries in datasets.items():
        eval_rows = []
        for sample in tqdm(dataset_entries):
            entities = None
            if c:
                doc = nlp(sample["text"])
                entities = [ent.text for ent in doc.ents]
                entities = ", ".join(entities)
            model_output = None
            if entities:
                model_output = transcribe(sample["audio"], prompt=entities)
            else:
                model_output = transcribe(sample["audio"])
            wer = jiwer.wer(normalize(sample["text"]), normalize(model_output))
            cer = jiwer.cer(normalize(sample["text"]), normalize(model_output))
            eval_rows.append({"audio_path": sample["audio"],
                              "label": normalize(sample["text"]),
                              "prediction": normalize(model_output),
                              "wer" : wer,
                              "cer": cer
                             }
                         )
        output_statistics = []
        results = pd.DataFrame(eval_rows)
        results.head()
        output_statistics.append(f"Mean WER: {results["wer"].mean()}")
        output_statistics.append(f"Mean CER: {results["cer"].mean()}")
        results = results.sort_values("wer", ascending=False)
        output_name = ""
        if c:
            output_name = f"qwen_{dataset_title}_with_context"
        else:
            output_name = f"qwen_{dataset_title}_without_context"
        results.to_csv(f"{output_name}.tsv", sep="\t", index=False)
        with open(f"{output_name}_statistics.txt", "w", encoding="UTF-8") as wf:
            for line in output_statistics:
                wf.write(line+"\n")
        print("="*50)
        print(dataset_title)
        print(output_name)
        print("\n".join(output_statistics))
        print("="*50)



print("Mean WER:", results["wer"].mean())
print("Mean CER:", results["cer"].mean())
results[["wer", "cer"]].describe()
results = results.sort_values("wer", ascending=False)
results.to_csv("results_with_context.tsv", sep="\t", index=False)