# Librerias Hugging faces
Primero creamos nuestro entorno virtual: \
`python3 -m venv .venv` \
<span style="color: green;">linux</span> \
`source .venv/bin/activate` \
<span style="color: red;">Windows</span> \
`.\.venv\Scripts\Activate.ps1` \
`.\.venv\Scripts\activate` \
`.venv\Scripts\activate.bat
<span style="color: blue;">linux&Windows</span> \
`pip install torch --index-url https://download.pytorch.org/whl/cpu` \
`pip install -U huggingface_hub diffusers transformers accelerate safetensors peft` \
Como correr prueba sola: \
`python3 sd15.py "description in english" -e estilo -s seed -p steps`\
usage: sd15.py [-h]
               [-e {caricatura,lapiz,acuarela,tinta,isometrico,vector,crochet,anime,pixel,comic,arcilla,synthwave,arcade}]
               [-p 1-10] [-s SEMILLA] [-o SALIDA] [--svg]
               prompt

Texto a imagen con SD 1.5 + LCM-LoRA

positional arguments:
  prompt                qué dibujar (en inglés funciona mejor)

options:
  -h, --help            show this help message and exit
  -e, --estilo {caricatura,lapiz,acuarela,tinta,isometrico,vector,crochet,anime,pixel,comic,arcilla,synthwave,arcade}
  -p, --pasos 1-10      pasos de difusión (default 4)
  -s, --semilla SEMILLA
                        semilla para repetir un resultado (default: aleatoria)
  -o, --salida SALIDA   carpeta de salida
  --svg                 intenta vectorizar con vtracer
