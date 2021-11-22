FROM python:3.8.5

RUN mkdir /code
COPY requirements.txt /code
RUN pip install -r /code/requirements.txt

COPY . /code
WORKDIR /code
RUN python manage.py makemigrations
RUN python manage.py migrate
RUN python manage.py dumpdata > dump.json
CMD gunicorn investor_assistant.wsgi:application --bind 0.0.0.0:8000