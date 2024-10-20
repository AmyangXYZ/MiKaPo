let wasm;

const cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : { decode: () => { throw Error('TextDecoder not available') } } );

if (typeof TextDecoder !== 'undefined') { cachedTextDecoder.decode(); };

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

let heap_next = heap.length;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function getObject(idx) { return heap[idx]; }

function isLikeNone(x) {
    return x === undefined || x === null;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder('utf-8') : { encode: () => { throw Error('TextEncoder not available') } } );

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
    return instance.ptr;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_exn_store(addHeapObject(e));
    }
}

const PoseSolverFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_posesolver_free(ptr >>> 0, 1));

export class PoseSolver {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PoseSolverFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_posesolver_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.posesolver_new();
        this.__wbg_ptr = ret >>> 0;
        PoseSolverFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Array<any>} main_body
     * @param {Array<any>} left_hand
     * @param {Array<any>} right_hand
     * @returns {PoseSolverResult}
     */
    solve(main_body, left_hand, right_hand) {
        const ret = wasm.posesolver_solve(this.__wbg_ptr, addHeapObject(main_body), addHeapObject(left_hand), addHeapObject(right_hand));
        return PoseSolverResult.__wrap(ret);
    }
}

const PoseSolverResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_posesolverresult_free(ptr >>> 0, 1));

export class PoseSolverResult {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(PoseSolverResult.prototype);
        obj.__wbg_ptr = ptr;
        PoseSolverResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PoseSolverResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_posesolverresult_free(ptr, 0);
    }
    /**
     * @returns {Rotation}
     */
    get upper_body() {
        const ret = wasm.__wbg_get_posesolverresult_upper_body(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set upper_body(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_upper_body(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get lower_body() {
        const ret = wasm.__wbg_get_posesolverresult_lower_body(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set lower_body(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_lower_body(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get neck() {
        const ret = wasm.__wbg_get_posesolverresult_neck(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set neck(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_neck(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get left_hip() {
        const ret = wasm.__wbg_get_posesolverresult_left_hip(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set left_hip(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_left_hip(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get right_hip() {
        const ret = wasm.__wbg_get_posesolverresult_right_hip(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set right_hip(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_right_hip(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get left_foot() {
        const ret = wasm.__wbg_get_posesolverresult_left_foot(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set left_foot(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_left_foot(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get right_foot() {
        const ret = wasm.__wbg_get_posesolverresult_right_foot(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set right_foot(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_right_foot(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get left_upper_arm() {
        const ret = wasm.__wbg_get_posesolverresult_left_upper_arm(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set left_upper_arm(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_left_upper_arm(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get right_upper_arm() {
        const ret = wasm.__wbg_get_posesolverresult_right_upper_arm(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set right_upper_arm(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_right_upper_arm(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get left_lower_arm() {
        const ret = wasm.__wbg_get_posesolverresult_left_lower_arm(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set left_lower_arm(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_left_lower_arm(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get right_lower_arm() {
        const ret = wasm.__wbg_get_posesolverresult_right_lower_arm(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set right_lower_arm(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_right_lower_arm(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get left_wrist() {
        const ret = wasm.__wbg_get_posesolverresult_left_wrist(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set left_wrist(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_left_wrist(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get right_wrist() {
        const ret = wasm.__wbg_get_posesolverresult_right_wrist(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set right_wrist(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_right_wrist(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get left_index_finger_mcp() {
        const ret = wasm.__wbg_get_posesolverresult_left_index_finger_mcp(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set left_index_finger_mcp(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_left_index_finger_mcp(this.__wbg_ptr, ptr0);
    }
    /**
     * @returns {Rotation}
     */
    get left_index_finger_pip() {
        const ret = wasm.__wbg_get_posesolverresult_left_index_finger_pip(this.__wbg_ptr);
        return Rotation.__wrap(ret);
    }
    /**
     * @param {Rotation} arg0
     */
    set left_index_finger_pip(arg0) {
        _assertClass(arg0, Rotation);
        var ptr0 = arg0.__destroy_into_raw();
        wasm.__wbg_set_posesolverresult_left_index_finger_pip(this.__wbg_ptr, ptr0);
    }
}

const RotationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_rotation_free(ptr >>> 0, 1));

export class Rotation {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Rotation.prototype);
        obj.__wbg_ptr = ptr;
        RotationFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RotationFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_rotation_free(ptr, 0);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    constructor(x, y, z, w) {
        const ret = wasm.rotation_new(x, y, z, w);
        this.__wbg_ptr = ret >>> 0;
        RotationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Rotation}
     */
    static get default() {
        const ret = wasm.rotation_default();
        return Rotation.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.rotation_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.rotation_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get z() {
        const ret = wasm.rotation_z(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get w() {
        const ret = wasm.rotation_w(this.__wbg_ptr);
        return ret;
    }
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                if (module.headers.get('Content-Type') != 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbindgen_string_new = function(arg0, arg1) {
        const ret = getStringFromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_number_get = function(arg0, arg1) {
        const obj = getObject(arg1);
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
        takeObject(arg0);
    };
    imports.wbg.__wbg_get_5419cf6b954aa11d = function(arg0, arg1) {
        const ret = getObject(arg0)[arg1 >>> 0];
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_length_f217bbbf7e8e4df4 = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_get_ef828680c64da212 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(getObject(arg0), getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbindgen_debug_string = function(arg0, arg1) {
        const ret = debugString(getObject(arg1));
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_throw = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };

    return imports;
}

function __wbg_init_memory(imports, memory) {

}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;



    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    __wbg_init_memory(imports);

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('pose_solver_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    __wbg_init_memory(imports);

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
