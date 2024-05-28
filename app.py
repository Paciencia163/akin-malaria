import streamlit as st
from tensorflow.keras.models import load_model
from files_upload import FilesUpload
import numpy as np

@st.cache_resource
def load_model_once():
    # Cachear o modelo para evitar recarregar a cada vez
    return load_model('./models/malaria_detector.h5')

def main():
    st.title("AKIN - Diagnósticos de Doenças")
    
    atividade = ['Predição de Malária', 'Predição de Febre Tifoide (Brevemente)']
    escolha = st.sidebar.selectbox('Escolha uma Atividade', atividade)

    if escolha == 'Predição de Malária':
        st.header("Predição de Malária")
        
        files_upload = FilesUpload()
        img = files_upload.run()

        if img is not None:
            st.image(img, caption='Imagem Carregada', use_column_width=True)

            if st.button("Prever"):
                st.text('Aguarde... O modelo está sendo carregado!')
                model = load_model_once()
                st.success("Modelo Carregado")
                st.text('Aguarde...')

                # Pré-processar a imagem para corresponder ao formato de entrada do modelo, se necessário
                if len(img.shape) == 4:  # Verifica se a imagem já tem dimensão de lote
                    img = np.squeeze(img, axis=0)  # Remove a dimensão extra do lote
                img = np.expand_dims(img, axis=0)  # Adiciona a dimensão do lote

                prediction = model.predict(img)[0][0]
                if prediction > 0.5:
                    st.markdown(
                        f"<h2 style='color: green;'>Não infectado</h2>"
                        f"<h3 style='color: green;'>Probabilidade: {prediction:.2f}</h3>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<h2 style='color: red;'>Infectado/Parasitado</h2>"
                        f"<h3 style='color: red;'>Probabilidade: {prediction:.2f}</h3>",
                        unsafe_allow_html=True
                    )

    elif escolha == 'Predição de Febre Tifoide (Brevemente)':
        st.header("Predição de Febre Tifoide")
        st.info("Esta funcionalidade estará disponível em breve. Fique atento para futuras atualizações.")

if __name__ == '__main__':
    main()
