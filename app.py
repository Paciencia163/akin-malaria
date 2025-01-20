import streamlit as st
from tensorflow.keras.models import load_model
from files_upload import FilesUpload
import numpy as np

# Função para carregar o modelo com cache
@st.cache_resource
def load_model_once():
    return load_model('./models/malaria_detector.h5')

# Função para pré-processar a imagem
def preprocess_image(img, image_shape=(130, 130, 3)):
    if len(img.shape) == 4:  # Verifica se a imagem já tem dimensão de lote
        img = np.squeeze(img, axis=0)  # Remove a dimensão extra do lote
    img = np.expand_dims(img, axis=0)  # Adiciona a dimensão do lote
    return img

# Função principal
def main():
    st.title("AKIN - Diagnósticos de Doenças")
    
    atividades = ['Predição de Malária', 'Predição de Febre Tifoide (Brevemente)']
    escolha = st.sidebar.selectbox('Escolha uma Atividade', atividades)

    if escolha == 'Predição de Malária':
        st.header("Predição de Malária")
        
        files_upload = FilesUpload()
        images = files_upload.run(max_files=20)

        if images:
            st.info(f"{len(images)} imagens carregadas com sucesso!")
            
            if st.button("Prever"):
                st.info("Carregando o modelo...")
                model = load_model_once()
                st.success("Modelo carregado!")

                predictions = []
                for i, img in enumerate(images):
                    img = preprocess_image(img)
                    prediction = model.predict(img)[0][0]
                    predictions.append(prediction)
                    label = "Não infectado" if prediction > 0.5 else "Infectado/Parasitado"
                    color = "green" if prediction > 0.5 else "red"
                    st.markdown(
                        f"<h3 style='color: {color};'>Imagem {i+1}: {label} (Probabilidade: {prediction:.2f})</h3>",
                        unsafe_allow_html=True,
                    )

                # Cálculo da média
                avg_prediction = np.mean(predictions)
                st.markdown(
                    f"<h2 style='color: blue;'>Média das Predições: {avg_prediction:.2f}</h2>",
                    unsafe_allow_html=True,
                )

    elif escolha == 'Predição de Febre Tifoide (Brevemente)':
        st.header("Predição de Febre Tifoide")
        st.info("Esta funcionalidade estará disponível em breve. Fique atento para futuras atualizações.")

if __name__ == '__main__':
    main()

