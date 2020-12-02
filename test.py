file = open('clean_text.txt', 'r')
data = file.readlines()
clean_text = []
for s in data:
    clean_text.append(s.rstrip('\n'))
file.close()

file = open('clean_summary.txt', 'r')
data = file.readlines()
clean_summary = []
for s in data:
    clean_text.append(s.rstrip('\n'))
file.close()

print(clean_text[:5], clean_summary[:5])
