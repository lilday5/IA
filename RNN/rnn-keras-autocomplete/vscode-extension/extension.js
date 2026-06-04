const vscode = require("vscode");
const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");

let proc, rl, pending = new Map(), reqId = 0;

function serverScript() {
  const cfg = vscode.workspace.getConfiguration("rnnKeras");
  const custom = cfg.get("serverScript");
  if (custom) return custom;
  const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  return path.join(root, "rnn-keras-autocomplete", "server_stdio.py");
}

function request(method, fields) {
  return new Promise((resolve, reject) => {
    if (!proc) {
      const py = vscode.workspace.getConfiguration("rnnKeras").get("pythonPath") || "python3";
      const script = serverScript();
      proc = spawn(py, [script], {
        cwd: path.dirname(script),
        stdio: ["pipe", "pipe", "pipe"]
      });
      proc.stderr.on("data", (d) => console.log("[RNN]", d.toString()));
      rl = readline.createInterface({ input: proc.stdout });
      rl.on("line", (line) => {
        try {
          const msg = JSON.parse(line);
          if (pending.has(msg._id)) {
            pending.get(msg._id)(msg);
            pending.delete(msg._id);
          }
        } catch (e) {
          console.error("[RNN] JSON parse error:", e);
        }
      });
      proc.on("exit", () => { proc = null; });
    }
    const id = ++reqId;
    const timer = setTimeout(() => reject(new Error("timeout")), 20000);
    pending.set(id, (msg) => {
      clearTimeout(timer);
      msg.ok ? resolve(msg) : reject(new Error(msg.error || "error"));
    });
    proc.stdin.write(JSON.stringify({ method, _id: id, ...fields }) + "\n");
  });
}

async function completeLine() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;
  const pos = ed.selection.active;
  const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
  if (!prefix.trim()) {
    vscode.window.showInformationMessage("Escribe algo antes de autocompletar.");
    return;
  }
  const maxNew = vscode.workspace.getConfiguration("rnnKeras").get("maxNew") || 60;
  try {
    vscode.window.showInformationMessage("RNN: completando...");
    const res = await request("complete", { prefix, max_new: maxNew, temperature: 0.75 });
    const suffix = res.text.slice(prefix.length).split("\n")[0];
    await ed.edit((eb) => eb.insert(pos, suffix));
  } catch (e) {
    vscode.window.showErrorMessage("RNN error: " + e.message);
  }
}

async function showSuggestions() {
  const ed = vscode.window.activeTextEditor;
  if (!ed) return;
  const pos = ed.selection.active;
  const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
  if (!prefix.trim()) {
    vscode.window.showInformationMessage("Escribe algo antes de pedir sugerencias.");
    return;
  }
  try {
    const res = await request("suggest", { prefix, n: 5 });
    const pick = await vscode.window.showQuickPick(res.items, { placeHolder: "Sugerencias RNN" });
    if (!pick) return;
    await ed.edit((eb) => eb.insert(pos, pick.slice(prefix.length)));
  } catch (e) {
    vscode.window.showErrorMessage("RNN error: " + e.message);
  }
}

function activate(ctx) {
  ctx.subscriptions.push(
    vscode.commands.registerCommand("rnnKeras.complete", completeLine),
    vscode.commands.registerCommand("rnnKeras.suggest", showSuggestions)
  );
  console.log("RNN Keras Autocomplete activado.");
}

function deactivate() {
  if (proc) proc.kill();
}

module.exports = { activate, deactivate };

/*function serverScript() {
  const cfg = vscode.workspace.getConfiguration("rnnKeras");
  const custom = cfg.get("serverScript");
  if (custom) {
    console.log("SERVER:", custom);
    return custom;
  }

  const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  const p = path.join(root, "rnn-keras-autocomplete", "server_stdio.py");

  console.log("SERVER:", p);

  return p;
}*/