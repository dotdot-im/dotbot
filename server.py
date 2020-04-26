import web
import json

from bot import bot

urls = (
    '/', 'interactive'
)
app = web.application(urls, globals())
bot = bot()

class interactive:
    def GET(self):
        data = web.input(persona=[], chat_history=[], reply=None)
        response = {
            'text': bot.get_text(data.persona, data.chat_history, data.reply)
        }

        web.header('Content-Type', 'application/json')
        return json.dumps(response)

web.webapi.internalerror = web.debugerror
if __name__ == '__main__': app.run()
