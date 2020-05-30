from flask import Flask, flash, redirect, render_template, request, session, abort
import Animation
import time
import prime

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('temp.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/send', methods = ['GET', 'POST'])
def send():
    if request.method == 'POST':
        name = request.form['name']
        name = str(name)
        now = time.strftime("%Y%m%d-%H%M%S")
        isin = Animation.isin(name)
        plottype = str(request.form.get('plot'))
        if isin == True:
            prime_years = prime.prime(name)
            if plottype == 'move':
                Animation.dynamic(name, now)
                filename = 'script' + now + '.gif'
                return render_template('send.html', playername = name, file = filename, years = prime_years)
            elif plottype == 'no-move':
                static = True
                Animation.static(name, now)
                filename = 'plot' + now + '.png'
                return render_template('send.html', playername = name, file = filename,  years = prime_years, static = static)
        else:
            return render_template('error.html')



if __name__ == "__main__":
    app.run()