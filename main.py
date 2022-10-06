import os
import streamlit as st
import soundfile as sf
from io import BytesIO
import numpy as np
import pydaisi as pyd
import librosa
import streamlit.components.v1 as components

whisper_model_gpu = pyd.Daisi("kanav/Whisper Model-GPU")

st.markdown("## Whisper Web-ui")
st.markdown("Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.")
st.markdown("Visit model api -> https://app.daisi.io/daisies/kanav/Whisper%20Model-GPU/api")

st.markdown("#### Upload your file ")
upload_wav = st.file_uploader("Upload a .wav sound file ",key="upload")

if upload_wav is not None:
            st.audio(upload_wav)
            btn = st.button("Generate")
            if btn:
                # print(data)h
                text = whisper_model_gpu.infer_wave_byte(upload_wav.getvalue()).value
                st.markdown(text)


st.markdown("#### Record your sound ")
st_audiorec = components.declare_component("st_audiorec", path=os.path.join(os.path.dirname(__file__), "st_audiorec/frontend/build"))
val = st_audiorec()

if isinstance(val, dict):  # retrieve audio data
    btn = st.button("Generate",key='12')
    if btn:
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            

            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            stream.seek(0)
            X, sample_rate = sf.read(stream)

            print(X,sample_rate)

            y_8k = librosa.resample( X[:,0], orig_sr=sample_rate, target_sr=16000)

            text = whisper_model_gpu.inference(y_8k.astype(np.float32) ).value
            st.markdown(text)