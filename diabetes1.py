import streamlit as st
import pickle
import numpy as np

#Exercício: Montar uma interface no Streamlit para classificar uma pessoa como portadora de diabetes ou não.
#O modelo de machine learning já está treinado e salvo em um arquivo chamado "trained_model.sav".

#os dados de entrada são:
#1. Número de vezes grávida
#2. Concentração de glicose
#3. Pressão sanguínea
#4. Espessura da pele
#5. Insulina
#6. IMC
#7. Função de pedigree de diabetes
#8. Idade

#todos esses dados são numéricos

#o input do modelo deve ser um array numpy 2d com todas features listadas acima nessa ordem

#o modelo deve retornar 0 ou 1
#se o resultado for 1, a pessoa é portadora de diabetes
#se o resultado for 0, a pessoa não é portadora de diabetes

st.title("Análise de Resultados Clínicos para Diabetes")
st.write("Atráves de um modelo de machine learning, o modelo irá realizar a ánalise dos dados para averiguar se o paciente possui diabetes")
st.subheader("Informe os dados solicitados abaixo:")




vezes_gravida = st.number_input("Passou por quanto períodos gestativos:",0,10)
glicose = st.number_input("Concentração de Glicose:",0,300)
blood_press = st.number_input("Pressão Sanguínea:",0,200)
espess_pele = st.number_input("Espessura da Pele:",0,100)
insulin = st.number_input("insulina:",0,1000)
imc = st.number_input("IMC:",value=0.0, format="%.2f")
diabetes_pedigree = st.number_input("Função de pedigree de diabetes:",value=0.0, format="%.2f")
idade = st.number_input("Idade:",0,122)


button = st.button("Continuar")
if button:
        input = np.array([[vezes_gravida, glicose, blood_press,espess_pele,insulin,imc,diabetes_pedigree,idade]])

def load_model():
   with open('trained_model.sav', 'rb') as file:
        model = pickle.load(file)
        return model


data = load_model()

prediction = data.predict(input) 
if prediction == 0:   
        st.title(f"Não há indicios de diabetes,Resultado da Análise:, {prediction}.")
else:
        st.title(f"Há indicios de diabetes,Resultado da Análise:, {prediction}.")
        


'''def main():
    
if __name__ == '__main__':
    main()'''