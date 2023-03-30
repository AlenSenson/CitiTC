
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline,TFAutoModelForQuestionAnswering
import tensorflow 


#summary
tokenizers = AutoTokenizer.from_pretrained('t5-base')
models = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)


#qa
modelq = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad",return_dict=False)
tokenizerq = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
nlp = pipeline("question-answering", model=modelq, tokenizer=tokenizerq)

############

sequence=input("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPaste T&C:\n")


res=0
while res!=3:
  print("\n\n\nMenu\n")
  print("1)Sumerise T&C\n")
  print("2)Ask Questions\n")
  print("3)Exit\n\n")

  res=int(input("Response:"))

  if res==1:
    
    inputs = tokenizers.encode("summarize: " + sequence,
                          return_tensors='pt',
                          max_length=512,
                          truncation=True)

    summary_ids = models.generate(inputs, max_length=200, min_length=1, length_penalty=5., num_beams=2)
  
    summary = tokenizers.decode(summary_ids[0])

    print("\n\n\n",summary,"\n\n\n")
  elif res==2:
    context= sequence
    question = input("Question:")

    result = nlp(question = question, context=context)

    #print (f"QUESTION: {question}") 
    print("\n\n\n")
    print(f"ANSWER: {result['answer']}")
    print("\n\n\n")
  elif res!=3:
    print("Invalid option\n\n\n")