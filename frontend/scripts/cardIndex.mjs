/* Settings sidebar sub-heading index — extracted from source, not hand-kept.

   The Settings page's desktop rail lists the top-level sections. This index
   adds the second level: every <Card> inside a section, in the order the user
   actually sees them, by their rendered <h2> title. That lets the rail
   hover-open a section and offer quick jumps into the cards below it.

   Cards are THE unit of visual grouping in settings: each section file renders
   a sequence of <Card>s (some directly, some via local card components like
   <KleinGenerationCard/>). Card takes an optional `id` (used by ?focus= deep
   links) and a `title` shown as the card's <h2>. Not every card carries an id,
   but for sidebar jumps EVERY card needs a scroll anchor — cards without a
   literal id get a stable slug id here (card-<slug(title)>), never persisted.

   ORDER: we follow render order from the main component's return, descending
   one level into local card components as they are referenced. A sidebar that
   lists cards in a different order than the page draws them is worse than no
   sidebar, so this walks the render tree rather than sorting alphabetically.

   Why extract from source instead of maintaining a parallel list: the sidebar
   must never drift from what actually renders. The contract test
   tests/cardIndex.contract.test.mjs re-runs this extraction and compares with
   the committed snapshot (src/components/settings/cardIndex.js), so the app
   ships whatever the test sees.

   Usage:
     node scripts/cardIndex.mjs            # print snapshot JS
     node scripts/cardIndex.mjs --check    # exit 1 if the snapshot is stale
*/
import { readdirSync, readFileSync, existsSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import path from 'node:path'

const SETTINGS_DIR = fileURLToPath(new URL('../src/components/settings/', import.meta.url))
const SNAPSHOT = fileURLToPath(new URL('../src/components/settings/cardIndex.js', import.meta.url))

const SECTION_BY_FILE = {
  OverviewSection: 'overview',
  EnginesSection: 'engines',
  ScrapingSection: 'scraping',
  LocalToolsSection: 'local-tools',
  CaptioningSection: 'captioning',
  TrainingSection: 'training',
  ServerSection: 'server',
  MaintenanceSection: 'maintenance',
}

/* Slugify a card title into a stable anchor id. */
export function slugTitle(title) {
  const base = title.toLowerCase()
    .replace(/&/g, 'and')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return base || 'card'
}

/* Find the char index just past a JSX opening tag's `>` starting at `start`.
   Honors quotes, backticks and {…} depth so multi-line props and template
   literals are skipped. Returns -1 if unbalanced. */
function findTagEnd(src, start) {
  const n = src.length
  let j = start
  let quote = null
  let brace = 0
  while (j < n) {
    const c = src[j]
    if (quote) {
      if (c === '\\') { j += 2; continue }
      if (c === quote) quote = null
      j += 1
      continue
    }
    if (brace > 0) {
      if (c === '`' || c === '"' || c === "'") { quote = c; j += 1; continue }
      if (c === '{') { brace += 1; j += 1; continue }
      if (c === '}') { brace -= 1; j += 1; continue }
      j += 1
      continue
    }
    if (c === '"' || c === "'" || c === '`') { quote = c; j += 1; continue }
    if (c === '{') { brace = 1; j += 1; continue }
    if (c === '>') return j + 1
    j += 1
  }
  return -1
}

/* Parse literal attribute values (double-quoted, or a plain-string backtick)
   out of a JSX open tag body already cut at the closing `>`. */
function attrsFrom(body) {
  const attrs = {}
  const re = /([A-Za-z][\w-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|\{([\s\S]*?)\})/g
  let m
  while ((m = re.exec(body))) {
    const name = m[1]
    if (m[2] !== undefined) { attrs[name] = m[2]; continue }
    if (m[3] !== undefined) { attrs[name] = m[3]; continue }
    const inner = (m[4] || '').trim()
    const lit = inner.match(/^`([^`]*)`$/)
    if (lit) attrs[name] = lit[1].replace(/\$\{[^}]*\}/g, '')
  }
  return attrs
}

/* The region of `src` that is the JSX render body of component `name`,
   measured from just after `return (`. Returns the slice bounds. */
function renderRegion(src, defStart) {
  // defStart is just past the `{` of `function Name(…) {`
  const returnIdx = src.indexOf('return', defStart)
  if (returnIdx === -1) return null
  let j = returnIdx + 6
  while (j < src.length && /\s/.test(src[j])) j += 1
  if (src[j] === '(') j += 1
  // balance parens from here
  let depth = 1
  let quote = null
  let brace = 0
  while (j < src.length) {
    const c = src[j]
    if (quote) {
      if (c === '\\') { j += 2; continue }
      if (c === quote) quote = null
      j += 1
      continue
    }
    if (brace > 0) {
      if (c === '`' || c === '"' || c === "'") { quote = c; j += 1; continue }
      if (c === '{') { brace += 1; j += 1; continue }
      if (c === '}') { brace -= 1; j += 1; continue }
      j += 1
      continue
    }
    if (c === '"' || c === "'" || c === '`') { quote = c; j += 1; continue }
    if (c === '{') { brace = 1; j += 1; continue }
    if (c === '(') { depth += 1; j += 1; continue }
    if (c === ')') {
      depth -= 1
      if (depth === 0) return { start: returnIdx, end: j }
      j += 1
      continue
    }
    j += 1
  }
  return null
}

/* Given a component's definition start, find where its function body ends
   (the balanced closing brace of `function Name(…) {`). */
/* Advance past a JS line/block comment at src[j] (src[j] must be '/').
   Returns the index just past the comment, or null when this is NOT a comment
   (a division, a regex literal…). Never called inside quotes/braces where a
   '{' would be balanced — only lexical comment skipping is needed. */
function skipJsComment(src, j) {
  const d = src[j + 1]
  if (d === '/') {
    const nl = src.indexOf('\n', j + 2)
    return nl === -1 ? src.length : nl + 1
  }
  if (d === '*') {
    const close = src.indexOf('*/', j + 2)
    return close === -1 ? src.length : close + 2
  }
  return null
}

/* Given a component's definition start, find where its function body ends
   (the balanced closing brace of `function Name(…) {`). */
function functionBodyEnd(src, defStart) {
  let j = defStart
  let depth = 1
  let quote = null
  let brace = 0
  while (j < src.length) {
    const c = src[j]
    // Comments never contribute { } to the balance; a // ZIP mode: {…} note
    // comment would otherwise derail the count for the whole rest of the file.
    if (!quote && c === '/') {
      const r = skipJsComment(src, j)
      if (r !== null) { j = r; continue }
    }
    if (quote) {
      if (c === '\\') { j += 2; continue }
      if (c === quote) quote = null
      j += 1
      continue
    }
    if (brace > 0) {
      if (c === '`' || c === '"' || c === "'") { quote = c; j += 1; continue }
      if (c === '{') { brace += 1; j += 1; continue }
      if (c === '}') { brace -= 1; j += 1; continue }
      j += 1
      continue
    }
    if (c === '"' || c === "'" || c === '`') { quote = c; j += 1; continue }
    if (c === '{') { brace = 1; j += 1; continue }
    if (c === '}') {
      depth -= 1
      if (depth === 0) return j
      j += 1
      continue
    }
    j += 1
  }
  return src.length
}

/* Walk a source slice collecting <Card>s and local card component calls in
   order. `cardNames`: set of local component names (function FooCard(…) in this
   file). `visited`: recursion guard. */
function walk(src, start, end, section, cardNames, visited, out) {
  let idx = start
  while (idx < end) {
    const cardAt = src.indexOf('<Card', idx)
    const cardWithin = cardAt !== -1 && cardAt < end
    // next local card component call inside the region
    let compAt = Infinity
    let compName = null
    for (const name of cardNames) {
      const at = src.indexOf(`<${name}`, idx)
      if (at !== -1 && at < end && at < compAt) { compAt = at; compName = name }
    }
    const compWithin = compAt !== Infinity
    if (!cardWithin && !compWithin) break
    const useComp = compWithin && (!cardWithin || compAt < cardAt)
    if (useComp) {
      const tagEnd = findTagEnd(src, compAt)
      if (tagEnd === -1) break
      if (visited.has(compName)) { idx = tagEnd; continue }
      visited.add(compName)
      const defRe = new RegExp(`function\\s+${compName}\\s*\\([^)]*\\)\\s*\\{`)
      const def = src.match(defRe)
      if (def) {
        const bodyStart = def.index + def[0].length
        const fEnd = functionBodyEnd(src, bodyStart)
        const reg = renderRegion(src, bodyStart)
        if (reg) walk(src, reg.start, fEnd, section, cardNames, visited, out)
        else walk(src, bodyStart, fEnd, section, cardNames, visited, out)
      }
      idx = tagEnd
      continue
    }
    // a <Card> (guard against <CardFoo)
    const after = src[cardAt + 5]
    if (after && /[A-Za-z0-9]/.test(after)) { idx = cardAt + 5; continue }
    const tagEnd = findTagEnd(src, cardAt)
    if (tagEnd === -1) break
    const body = src.substring(cardAt, tagEnd)
    const attrs = attrsFrom(body)
    if (attrs.title) {
      out.push({
        section,
        title: attrs.title,
        id: attrs.id || `card-${slugTitle(attrs.title)}`,
      })
    }
    idx = tagEnd
  }
}

/* Extract the cards of one section, in render order. */
function cardsFromFile(src, section) {
  const localCardNames = new Set(
    [...src.matchAll(/function\s+([A-Z][A-Za-z0-9]*Card)\s*\(/g)].map((m) => m[1]),
  )
  const main = src.match(/export\s+default\s+function\s+([A-Za-z0-9]+)Section\s*\([^)]*\)\s*\{/)
  if (!main) return []
  const bodyStart = main.index + main[0].length
  const fEnd = functionBodyEnd(src, bodyStart)
  const out = []
  const visited = new Set()
  const reg = renderRegion(src, bodyStart)
  if (reg) {
    // scan from the main return to the end of the function body — the return
    // region's own closing paren is not reliably identifiable (nested arrow
    // functions inside JSX attrs break paren-balancing), but the function
    // body's closing brace is.
    walk(src, reg.start, fEnd, section, localCardNames, visited, out)
  } else {
    walk(src, bodyStart, fEnd, section, localCardNames, visited, out)
  }
  return out
}

export function extractCardIndex() {
  const files = readdirSync(SETTINGS_DIR)
    .filter((f) => f.endsWith('Section.jsx') && !f.endsWith('.test.jsx'))
    .sort()
  const out = []
  for (const f of files) {
    const base = f.replace(/\.jsx$/, '')
    const section = SECTION_BY_FILE[base]
    if (!section) continue
    const src = readFileSync(path.join(SETTINGS_DIR, f), 'utf8')
    out.push(...cardsFromFile(src, section))
  }
  return out
}

export function renderSnapshot(cards) {
  return `/* GENERATED by scripts/cardIndex.mjs — do not edit by hand.

   Settings sidebar sub-headings: one entry per <Card> rendered by the settings
   sections, in render order. Regenerate with:

     node scripts/cardIndex.mjs > src/components/settings/cardIndex.js

   The contract test (tests/cardIndex.contract.test.mjs) fails when this drifts
   from the source, so the app always ships what the rail can jump to. */
export const CARD_INDEX = ${JSON.stringify(cards, null, 2)}
`
}

const main = () => {
  const cards = extractCardIndex()
  const json = renderSnapshot(cards)
  if (process.argv.includes('--check')) {
    if (existsSync(SNAPSHOT) && readFileSync(SNAPSHOT, 'utf8').trim() === json.trim()) {
      console.log('cardIndex: snapshot is up to date')
      return
    }
    console.error('cardIndex: snapshot is STALE — run `node scripts/cardIndex.mjs > src/components/settings/cardIndex.js`')
    process.exit(1)
  }
  process.stdout.write(json)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main()
}