import re

file_path = 'c:/Users/adity/Desktop/dashboard/index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Remove inline transition-delays
html = re.sub(r' style="transition-delay:\s*\d+ms;"', '', html)

# Replace reveal-element in pipeline with architecture-node
def repl(m):
    return m.group(0).replace('reveal-element', 'architecture-node')

html = re.sub(r'(<div class="node[^>]*>|<div class="arrow[^>]*>|<div class="node-wrapper[^>]*>)', repl, html)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(html)
print('Done!')
