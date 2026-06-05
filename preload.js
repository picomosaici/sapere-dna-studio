const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("nativeAPI", {
  saveFile: (suggestedName, content, filters) =>
    ipcRenderer.invoke("save-file", { suggestedName, content, filters }),
  openText: () => ipcRenderer.invoke("open-text"),
  storeGet: (key) => ipcRenderer.invoke("store-get", key),
  storeSet: (key, value) => ipcRenderer.invoke("store-set", { key, value })
});
