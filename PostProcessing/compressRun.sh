runFolder=$(find . -maxdepth 1 -type d -name "*_${1}_*")
runTarName="run${1}.tar"
echo "${1}"
echo "${runFolder}"
tar -cf runTarName runFolder