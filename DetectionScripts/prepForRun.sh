if [ ! -d "fulloutput" ]; then
  mkdir "fulloutput"
fi

cd "fulloutput"

if [ ! -d "run${1}" ]; then
  mkdir "run${1}"
fi

cd ..

if [ ! -d "run${1}JobFiles" ]; then
  mkdir "run${1}JobFiles"
fi

tar -xf "${2}"

cp "${HOME}/templates/*" "run${1}JobFiles"
python3 "run${1}JobFiles/jobmaker.py ${1} ${2} ${3}"