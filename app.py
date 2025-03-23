from flask import Flask, render_template, request
import torch
from torch.nn import Module
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

app = Flask(__name__)

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("./saved_tokenizer")
teacher_model = AutoModelForSequenceClassification.from_pretrained("./exported_teacher_model")


teacher_config_dict = teacher_model.config.to_dict()
teacher_config_dict['num_hidden_layers'] //= 2
student_config = BertConfig.from_dict(teacher_config_dict)


model = type(teacher_model)(student_config)

def distill_bert_weights_odd(teacher: Module, student: Module) -> Module:
    """
    Recursively copies the teacher's weights into the student model,
    only copying odd-numbered layers from the teacher's encoder.
    """
    if isinstance(teacher, BertModel) or type(teacher).__name__.startswith('BertFor'):
        # Iterate through submodules
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_bert_weights_odd(teacher_part, student_part)

    elif isinstance(teacher, BertEncoder):
        # teacher has 12 layers, student has 6
        teacher_layers = [layer for layer in next(teacher.children())]  # 12 layers
        student_layers = [layer for layer in next(student.children())]  # 6 layers
        
        odd_layer_indices = [0, 2, 4, 6, 8, 10]  # Indices of "odd" layers to copy
        print("Using Odd Layers from Teacher Encoder")

        for i, idx in enumerate(odd_layer_indices):
            student_layers[i].load_state_dict(teacher_layers[idx].state_dict())

    else:
        student.load_state_dict(teacher.state_dict())

    return student


model = distill_bert_weights_odd(teacher=teacher_model, student=model)


model.load_state_dict(torch.load('./model/distil_odd_model.pt', map_location=device))

model.eval()
model.to(device)


label_mapping = {0: 'noHate', 1: 'hate', 2: 'idk/skip', 3: 'relation'}

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    prompt = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        if prompt:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            
            # Pick the highest logit
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            response = label_mapping[predicted_class_id]
            print("Predicted class ID:", predicted_class_id)

    return render_template("index.html", prompt=prompt, response=response)

if __name__ == "__main__":
    app.run(debug=True)
