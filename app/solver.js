const fs = require('fs');
const path = require('path');

const wasmPath = path.join(__dirname, 'sha3_wasm_bg.wasm');
const wasmBuf = fs.readFileSync(wasmPath);

let wasmInstance = null;

async function getInstance() {
    if (!wasmInstance) {
        const wasmModule = await WebAssembly.compile(wasmBuf);
        wasmInstance = await WebAssembly.instantiate(wasmModule, {});
    }
    return wasmInstance;
}

async function solve(cfg) {
    const instance = await getInstance();
    const prefix = cfg.salt + '_' + cfg.expire_at + '_';

    function writeStr(s) {
        const buf = Buffer.from(s, 'utf8');
        const ptr = instance.exports.__wbindgen_export_0(buf.length, 1);
        new Uint8Array(instance.exports.memory.buffer).set(buf, ptr);
        return { ptr, len: buf.length };
    }

    const retptr = instance.exports.__wbindgen_add_to_stack_pointer(-16);
    const ch = writeStr(cfg.challenge);
    const pfx = writeStr(prefix);
    instance.exports.wasm_solve(retptr, ch.ptr, ch.len, pfx.ptr, pfx.len, cfg.difficulty);
    const val = new Float64Array(instance.exports.memory.buffer)[(retptr + 8) / 8];
    instance.exports.__wbindgen_add_to_stack_pointer(16);
    return Math.floor(val);
}

if (require.main === module) {
    const cfg = JSON.parse(process.argv[2]);
    solve(cfg).then(ans => {
        console.log(ans);
    }).catch(err => {
        console.error(err);
        process.exit(1);
    });
}

module.exports = { solve };
