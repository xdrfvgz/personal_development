#!/usr/bin/env python3

import os
import json
import subprocess
import datetime
from typing import List, Optional
from dataclasses import dataclass
import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

@dataclass
class Message:
    role: str
    content: str

class CLIAssistant:
    def __init__(self):
        self.console = Console()
        self.client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = "claude-3-5-sonnet-20241022"
        self.conversation_history: List[Message] = []
        
        self.system_prompt = """# System-Prompt: KI-Assistent für Kommandozeilen-Interaktion

Sie sind ein KI-Assistent mit Zugriff auf die Kommandozeile des Benutzers.
Befolgen Sie diese Richtlinien:

1. Befehlsausführung:
   - Umschließen Sie Befehle mit [cmd] und [/cmd] Tags.
   - Beispiel: [cmd]mkdir neuer_ordner[/cmd]

2. Rückmeldung:
   - Sie erhalten die Befehlsausgabe in der nächsten Benutzeranfrage.
   - Analysieren Sie diese für weitere Aktionen.

3. Sicherheit:
   - Seien Sie vorsichtig mit systemverändernden Befehlen.
   - Fragen Sie bei riskanten Aktionen nach Bestätigung.

4. Erklärungen:
   - Erläutern Sie kurz die Wirkung jedes Befehls."""

    def print_error(self, message: str):
        """Gibt eine Fehlermeldung sicher aus."""
        error_text = Text()
        error_text.append("Fehler: ", style="bold red")
        error_text.append(message, style="red")
        self.console.print(error_text)

    def print_labeled(self, label: str, content: str, label_style: str = "bold blue"):
        """Gibt formatierten Text mit Label aus."""
        self.console.print()  # Leerzeile
        self.console.print(Text(label, style=label_style))
        self.console.print(content)

    def add_to_history(self, role: str, content: str):
        """Fügt eine neue Nachricht zur Konversationshistorie hinzu."""
        self.conversation_history.append(Message(role=role, content=content))

    def extract_command(self, text: str) -> Optional[str]:
        """Extrahiert den Befehl zwischen [cmd] und [/cmd] Tags."""
        try:
            start = text.index("[cmd]") + 5
            end = text.index("[/cmd]")
            return text[start:end].strip()
        except ValueError:
            return None

    def execute_command(self, command: str) -> str:
        """Führt einen Befehl direkt aus."""
        if not command:
            return "Kein gültiger Befehl gefunden."

        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            if not output and result.returncode != 0:
                return f"Befehl fehlgeschlagen mit Exit-Code {result.returncode}"
            return output if output else "Befehl ausgeführt (keine Ausgabe)"
            
        except subprocess.TimeoutExpired:
            return "Befehl wurde wegen Zeitüberschreitung (30s) abgebrochen"
        except Exception as e:
            return f"Fehler bei der Befehlsausführung: {str(e)}"

    def save_conversation(self):
        """Speichert die aktuelle Konversation in einer JSON-Datei."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "conversation": [
                    {"role": msg.role, "content": msg.content}
                    for msg in self.conversation_history
                ]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.console.print(Text(f"Konversation wurde in {filename} gespeichert.", style="green"))
        except Exception as e:
            self.print_error(f"Fehler beim Speichern der Konversation: {str(e)}")

    def load_conversation(self, filename: str):
        """Lädt eine Konversation aus einer JSON-Datei."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversation_history = [
                Message(role=msg["role"], content=msg["content"])
                for msg in data["conversation"]
            ]
            self.console.print(Text(f"Konversation aus {filename} geladen.", style="green"))
        except Exception as e:
            self.print_error(f"Fehler beim Laden der Konversation: {str(e)}")

    def show_history(self):
        """Zeigt die Konversationshistorie an."""
        if not self.conversation_history:
            self.console.print(Text("Keine Konversationshistorie vorhanden.", style="yellow"))
            return

        for msg in self.conversation_history:
            style = "blue" if msg.role == "assistant" else "green"
            self.console.print()
            self.console.print(Text(f"{msg.role}:", style=style))
            self.console.print(Panel(msg.content))

    def get_response(self, user_input: str) -> str:
        """Holt eine Antwort von der Anthropic API."""
        try:
            messages = [{"role": msg.role, "content": msg.content} 
                       for msg in self.conversation_history]
            messages.append({"role": "user", "content": user_input})

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=messages
            )

            return response.content[0].text
        except Exception as e:
            self.console.print_exception()
            return f"Fehler bei der API-Anfrage: {str(e)}"

    def run(self):
        """Hauptschleife des Assistenten."""
        self.console.print(Panel.fit(
            "Kommandozeilen-Assistent\n"
            "Befehle: exit, save, load <datei>, history"
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]>[/bold green]")

                if not user_input.strip():
                    continue

                if user_input == "exit":
                    self.console.print(Text("Auf Wiedersehen!", style="yellow"))
                    break
                elif user_input == "save":
                    self.save_conversation()
                    continue
                elif user_input == "history":
                    self.show_history()
                    continue
                elif user_input.startswith("load "):
                    self.load_conversation(user_input[5:].strip())
                    continue

                # Füge Benutzereingabe zur Historie hinzu
                self.add_to_history("user", user_input)

                # Hole Antwort von der API
                response = self.get_response(user_input)
                self.print_labeled("Assistant:", Markdown(response))
                self.add_to_history("assistant", response)

                # Extrahiere und führe Befehl aus
                if command := self.extract_command(response):
                    self.print_labeled("Befehl:", command, "bold yellow")
                    output = self.execute_command(command)
                    self.print_labeled("Ausgabe:", output, "bold yellow")
                    self.add_to_history("user", f"Befehl ausgeführt: {command}\nAusgabe: {output}")

            except KeyboardInterrupt:
                self.console.print(Text("\nBeenden mit 'exit'", style="yellow"))
            except Exception as e:
                self.print_error(str(e))
                self.console.print_exception()

def main():
    """Hauptfunktion zum Starten des Assistenten."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Fehler: ANTHROPIC_API_KEY Umgebungsvariable nicht gesetzt")
        return

    assistant = CLIAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
