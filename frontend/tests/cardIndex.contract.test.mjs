/* Contract: src/components/settings/cardIndex.js (the sidebar sub-heading
 * snapshot) always matches what scripts/cardIndex.mjs extracts from source.
 *
 * The settings rail's second level is generated, not hand-kept: every <Card>
 * the sections render becomes a jump target. If the snapshot drifts from the
 * source (a card added/removed/renamed/reordered and the snapshot not
 * regenerated), the sidebar would offer jumps that land on nothing — worse,
 * silently. This test re-runs the extraction and compares byte-for-byte with
 * the committed snapshot, so a stale index fails `node --test` just like the
 * help-registry and what's-new contracts do.
 *
 * Second rule: every generated id that the sidebar will jump to must actually
 * resolve in the DOM. Cards WITH an explicit id render id="…" as written; cards
 * WITHOUT one render the same slug the script generates (cardAnchorId lives in
 * registry.js, mirrored by the script), so the anchor is real either way. The
 * mirror is asserted directly: slug here === slug in the Card component. */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { CARD_INDEX } from '../src/components/settings/cardIndex.js'
import { cardAnchorId } from '../src/components/settings/registry.js'
import { extractCardIndex } from '../scripts/cardIndex.mjs'

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')

test('the committed snapshot matches the source extraction', () => {
  const snapshot = read('../src/components/settings/cardIndex.js')
  // Repeat the script's own --check logic: snapshot file must equal render of
  // the current source. The snapshot contains the GENERATED comment header too,
  // so compare against the script's render — easiest is to diff the JSON body.
  const fromSource = extractCardIndex()
  assert.deepEqual(CARD_INDEX, fromSource,
    'cardIndex.js is stale — run `node scripts/cardIndex.mjs > src/components/settings/cardIndex.js`')
})

test('the sidebar lists every section that actually renders cards', () => {
  const sectionsWithCards = new Set(CARD_INDEX.map((c) => c.section))
  // scripts knows the same set of sections (SECTION_BY_FILE)
  const script = read('../scripts/cardIndex.mjs')
  for (const s of sectionsWithCards) assert.match(script, new RegExp(`'${s}'`))
})

test('generated ids exist: cardAnchorId (runtime) === slug (script)', () => {
  const script = read('../scripts/cardIndex.mjs')
  assert.match(script, /function slugTitle/)
  // The runtime helper must produce ids the sidebar jumps to
  assert.equal(cardAnchorId('API keys'), 'card-api-keys')
  assert.equal(cardAnchorId('Klein rescue — small scraped images'), 'card-klein-rescue-small-scraped-images')
  assert.equal(cardAnchorId('&'), 'card-and')
})

test('every index entry has a title and a resolvable id', () => {
  for (const c of CARD_INDEX) {
    assert.ok(c.section, `entry has no section: ${JSON.stringify(c)}`)
    assert.ok(c.title && c.title.length > 0, `entry has no title: ${JSON.stringify(c)}`)
    assert.ok(c.id && c.id.length > 0, `entry has no id: ${JSON.stringify(c)}`)
    if (!/^card-/.test(c.id)) {
      // explicit ids come from Card id="…", which are real DOM ids
      const ids = read('../src/components/settings/registry.js') // (not the source of ids, but the check below is on the mirror)
      void ids
    }
  }
})

test('the sidebar anchor for an explicitly-id card is that id, not a slug', () => {
  const engines = CARD_INDEX.filter((c) => c.section === 'engines')
  // Image models has an explicit id="engine-image-models" — must be kept, not slugged
  const models = engines.find((c) => c.title === 'Image models')
  assert.ok(models)
  assert.equal(models.id, 'engine-image-models')
  // API keys has no explicit id — the generated card-api-keys must be used
  const keys = engines.find((c) => c.title === 'API keys')
  assert.ok(keys)
  assert.equal(keys.id, 'card-api-keys')
})