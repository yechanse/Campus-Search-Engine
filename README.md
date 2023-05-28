# Mini Search Engine
* For the search engine, I managed the parsing data through data pre-processing into pickle (.pkl) files into "cache." 
* I processed crawling, indexing, retrieval, and concepts of classification and clustering of text documents with local json files that parsed the UCI website, which are >10 GB size. 
* The main essence of this search engine is designing efficient data structures, devising efficient file access, and balancing memory us-age and response time.


## How to run this search engine (Local GUI)
1. Make sure you have downloaded the “DEV” folder and make a directory called “cache” where you would store all partial files and indexing data
2. Run “InvertedIndex.py”
3. The file named “inverted_index.txt” will be generated after full execution of the program. (this will probably take 15~50 minute depending on your system)
4. Run “search_gui.py”
5. Tkinter window will pop up and type the desired query in the entry box.
6. Click the ‘show the result’ to see the result of your query.
7. Exit by pressing the ‘esc’ button or close the window by mouse click.

*If you want to run the program on the console, follow the instructions until 3. and run “search.py” instead. You can now follow the instructions printed out in the console window.


## How to create custom index
1. Place the json files in the folder named "DEV" that you have already created above.
2. Set desired "counterLimit". Default is 10000. If you have a lot of memory, you can increase the limit. (less partial files will be created and therefore take less time to build the inverted_index.txt)
3. Run "InvertedIndex.py"
4. 'Partial_X.txt' files and three .pkl files will be generated in the ./cache directory.
    a. Partial files are created to prevent out of memory error (later merged as 'inverted_index.txt').
5. 'inverted_index.txt' will be generated in the directory that "InvertedIndex.py" is placed in.
6. Three .pkl files in the /cache folder and "inverted_index.txt" placed in the main directory are necessary to execute 'search.py' later.
