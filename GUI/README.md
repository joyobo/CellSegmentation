## Run User Interface (UI): 
1. Set-up for the UI locally.    
a. Create a virtual environment using: <br> `python -m venv venv`   
b. Activate the virtual environment:     
&nbsp; - Activate the virtualenv (OS X & Linux)   <br>`venv/bin/activate`   
&nbsp; - Activate the virtualenv (Windows) <br>  `venv\Scripts\activate`   
c. Install the necessary libraries:  
&nbsp; - Navigate out to the *app* folder. In the *app* folder  <br> `pip install -r requirements.txt`

2. Run the application using:   
   `python app.py`
   
## Using the UI
1. Go to http://127.0.0.1:5000/. 
2. Click Choose File and select the histology image file (in .png format) from your file explorer.
3. Click Upload. 
4. Similarly, click Choose File and select the corresponding label file (in .mat format, with the ground truth label stored as key ‘type_map’) from your file explorer.
5. Click Upload.
6. See the resultant mask and accuracy under the Results section.
