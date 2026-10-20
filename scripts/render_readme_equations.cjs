#!/usr/bin/env node
/** Render standalone LaTeX fragments as GitHub-compatible static SVGs.
 *
 * One-time tooling install, separate from runtime dependencies:
 *   npm install --prefix /tmp/causalbridge-mathjax-render mathjax-full@3.2.2
 * Render selected files (omit arguments to render all docs/equations/*.tex):
 *   NODE_PATH=/tmp/causalbridge-mathjax-render/node_modules \
 *     node scripts/render_readme_equations.cjs docs/equations/binning-4-01.tex
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document('', {
  InputJax: new TeX({packages: AllPackages}),
  OutputJax: new SVG({fontCache: 'none'}),
});
const directory = path.resolve(__dirname, '../docs/equations');
const requested = process.argv.slice(2);
const files = requested.length ? requested.map(file => path.resolve(file)) :
  fs.readdirSync(directory).filter(file => file.endsWith('.tex')).sort().map(file => path.join(directory, file));
for (const file of files) {
  if (!file.endsWith('.tex')) throw new Error(`Expected a .tex fragment: ${file}`);
  const node = document.convert(fs.readFileSync(file, 'utf8').trim(), {display: true});
  let svg = adaptor.innerHTML(node);
  if (/data-mjx-error|data-mml-node="merror"/.test(svg)) {
    throw new Error(`Invalid LaTeX in ${file}`);
  }
  // Match the existing section-6 SVG font, paths, size and foreground color.
  svg = svg.replace('<svg ', '<svg color="#24292f" ');
  const output = file.replace(/\.tex$/, '.svg');
  fs.writeFileSync(output, svg + '\n');
  process.stdout.write(`${path.relative(process.cwd(), output)}\n`);
}
