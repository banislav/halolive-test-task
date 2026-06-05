export function relTime(ts) {
  const delta = Date.now() - new Date(ts).getTime();
  const minutes = Math.floor(delta / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function markdownToHtml(text) {
  if (!text) return '';
  const formatInline = (value) =>
    escapeHtml(value)
      .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`\n]+)`/g, '<code>$1</code>');
  const lines = text.split('\n');
  let html = '';
  let index = 0;
  while (index < lines.length) {
    const trimmed = lines[index].trim();
    if (!trimmed) {
      index += 1;
      continue;
    }
    if (/^[-*]\s/.test(trimmed)) {
      html += '<ul>';
      while (index < lines.length && /^[-*]\s/.test(lines[index].trim())) {
        html += `<li>${formatInline(lines[index].trim().replace(/^[-*]\s+/, ''))}</li>`;
        index += 1;
      }
      html += '</ul>';
      continue;
    }
    let paragraph = trimmed;
    index += 1;
    while (index < lines.length) {
      const next = lines[index].trim();
      if (!next || /^[-*]\s/.test(next)) break;
      paragraph += ` ${next}`;
      index += 1;
    }
    html += `<p>${formatInline(paragraph)}</p>`;
  }
  return html;
}

export function jsonHtml(value) {
  return escapeHtml(JSON.stringify(value, null, 2)).replace(
    /("(?:\\.|[^"\\])*"(?:\s*:)?|\b(?:true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    (match) => {
      if (/^".*"\s*:$/.test(match.trim())) return `<span class="jk">${match}</span>`;
      if (/^"/.test(match)) return `<span class="js">${match}</span>`;
      if (/true|false/.test(match)) return `<span class="jb">${match}</span>`;
      if (match === 'null') return `<span class="jn">${match}</span>`;
      return `<span class="ji">${match}</span>`;
    },
  );
}
