# 🧠 Advanced Named Entity Recognition (NER) App

This project is a Named Entity Recognition (NER) application built using Streamlit, HuggingFace Transformers, and a fine-tuned BERT-based model. The app allows users to input text and receive a visual and tabular breakdown of detected named entities such as people, organizations, locations, and more.

It is designed to be user-friendly, interactive, and visually intuitive, making it easy for both technical and non-technical users to explore the power of natural language processing (NLP). The model highlights entities in real-time using color-coded tags and provides a summary table with confidence scores, entity types, and token positions.

![Dataset Link](https://huggingface.co/datasets/eriktks/conll2003)

---

## 🚀 Features

- ✅ Uses a fine-tuned BERT model for token classification.
- ✅ Entity-level color highlighting on input text.
- ✅ Summary table of all detected entities.
- ✅ Clean, responsive UI with Streamlit.
- ✅ Sidebar legend for label reference.
- ✅ Automatically maps model's numeric labels (e.g. '1', '2') to meaningful tags like `B-PER`, `I-ORG`, etc.

---

## 🧪 Label Legend

| Label ID | Meaning        | Description                         |
|----------|----------------|-------------------------------------|
| `'1'`    | `B-PER`        | Beginning of a person name          |
| `'2'`    | `I-PER`        | Inside of a person name             |
| `'3'`    | `B-ORG`        | Beginning of an organization        |
| `'4'`    | `I-ORG`        | Inside of an organization           |
| `'5'`    | `B-LOC`        | Beginning of a location             |
| `'6'`    | `I-LOC`        | Inside of a location                |
| `'7'`    | `B-MISC`       | Beginning of a miscellaneous entity |
| `'8'`    | `I-MISC`       | Inside of a miscellaneous entity    |

---

## 💡 Example Inputs

Try these in the app for testing:

- *"Elon Musk visited Tesla's headquarters in California on March 10, 2022."*
- *"Google was founded by Larry Page and Sergey Brin in September 1998."*
- *"The Eiffel Tower is a landmark in Paris, France."*

---

## Screenshot

![Image](https://github.com/user-attachments/assets/14d55e6e-8266-448f-9d95-5ab255cc00b1)
![Image](https://github.com/user-attachments/assets/177cceb4-8bfc-497a-a959-888e9fbb1716)

---

## 🧠 Model Info

- Base Model: `bert-base-uncased` (or your chosen architecture)
- Fine-tuned for: Named Entity Recognition
- Supported Entities: `PER`, `ORG`, `LOC`, `MISC`

---

## 📌 Notes

- Make sure your model uses the label scheme matching the ID map in the app.
- You can modify color schemes and label mappings in `app.py`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Muhammad Hamza**  
Machine Learning Engineer  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammad-hamza-khattak/) 
