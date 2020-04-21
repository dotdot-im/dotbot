import web
import json

urls = (
    '/', 'interactive'
)
app = web.application(urls, globals())

class interactive:
    def GET(self):
        user_data = web.input(id="no data")
        response = {
            'user_id': user_data.id,
        }

        web.header('Content-Type', 'application/json')
        return json.dumps(response)

web.webapi.internalerror = web.debugerror
if __name__ == '__main__': app.run()
