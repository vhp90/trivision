if (typeof globalThis.self === 'undefined') {
  Reflect.set(globalThis, 'self', globalThis);
}
