for f in *.py; do
  echo "===== $f =====" >> out.txt
  cat "$f" >> out.txt
  echo >> out.txt
done
