for f in *.py; do
  echo "===== FILE: $f ====="
  cat "$f"
  echo
done > output/all_files.txt
