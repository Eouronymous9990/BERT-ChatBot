import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from PIL import Image
import base64

model2 = BertForQuestionAnswering.from_pretrained(r"E:\NLPPPPPP\New folder (3)\folder")
tokenizer2 = BertTokenizer.from_pretrained(r"E:\NLPPPPPP\New folder (3)\folder2")

def get_answer(question, context):
    inputs = tokenizer2.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs['input_ids'][0].tolist()

    with torch.no_grad():
        outputs = model2(**inputs)
        start_scores, end_scores = outputs.start_logits, outputs.end_logits

    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1

    answer_tokens = input_ids[start_idx:end_idx]
    answer = tokenizer2.decode(answer_tokens)

    return answer

def load_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# image_url = 'https://www.example.com/your-image.jpg'

image_path = r"E:\NLPPPPPP\New folder (3)\BG.jpg"
image_base64 = load_image(image_path)

st.markdown(f"""
    <style>
        body {{
            background-image: url('data:image/jpg;base64,{image_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;  /*  */
        }}
        .streamlit-expanderHeader {{
            font-size: 20px;
        }}
        .css-ffhzg2 {{
            background-color: rgba(0, 0, 0, 0.5);  /* إضافة شفافية داكنة للخلفية */
        }}
        .css-1y5v7g8 {{
            background-color: rgba(0, 0, 0, 0.5); /* طبقة داكنة أسفل النص */
            padding: 20px;
            border-radius: 10px;
        }}
    </style>
""", unsafe_allow_html=True)

st.title("Question Answering with BERT🫠")
st.write("Enter the context and the question below to get the answer using a pre-trained BERT model.")

context = st.text_area("Context", "Enter the text that contains the information for the question.")
question = st.text_input("Question", "Enter the question you want to ask.")

if st.button("Submit"):
    if context and question:
        answer = get_answer(question, context)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter both context and question.")
