# Muda o bashrc
VAL="PS1='\[\033[01;32m\]\u\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\] \[\033[01;33m\]➜\[\033[00m\] '" python3 -c 'import os; f=os.path.expanduser("~/.bashrc"); l=open(f).readlines(); l[59]=os.environ["VAL"]+"\n"; open(f,"w").writelines(l)'
source ~/.bashrc

# faz pasta source
mkdir -pv ~/source && cd ~/source

git clone https://github.com/umjourje/Tupa-Masters.git && cd Tupa-Masters


######## Dentro do SUDO
# Verifica a versão do pip e instala o env exato (ex.: se fosse python3.14, seria python3.14-venv)
sudo su
PY_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
apt update && apt install "python${PY_VER}-venv" -y
######## Dentro do SUDO
# Aqui tem q sair do sudo


# Cria o virtualenv
PY_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2 | tr -d '.')
python3 -m venv "~/source/${PY_VER}-env"
source "~/source/${PY_VER}-env/bin/activate"

# Instala as libs necessárias para o contexto
pip install --upgrade pip
pip install -r requirements_pi.txt

# Verifica libs instaladas
python3 check_pi_config.py