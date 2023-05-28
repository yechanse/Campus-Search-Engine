from tkinter import *
from tkinter import messagebox
import search

wd = Tk()
wd.title('TEAM KAKAO')
wd.geometry("500x400")

origin = StringVar()  # Holds a string; default value ""


def calculate():
    ret = []


    query = origin.get() # 여기서 쿼리 받기.
    if query.strip() == "":
        messagebox.showwarning(title="Invalid Query", message="Please input valid queries")
        return

    se = search.SearchEngine() 
    se.loadIndexingData()

    se.processQuery(query)

    for docID in se.champion[:10]:
       ret.append(se.urlMap[docID])
       print(se.urlMap[docID])  # 혹여나 들어가라고 시킬수도 있으니깐 console에도 출력되게

    result_str = ''
    for i, url in enumerate(ret):
        result_str += (f'{i+1}.' + url + '\n')
        
    if result_str != '':
        messagebox.showinfo('RESULT', result_str)
    else:
        messagebox.showwarning(title="No match", message="No matches found for your query.")
        return

def exit(event):
    wd.destroy()


if __name__ == "__main__":
    # 위젯 생성
    lb0 = Label(wd, text='Team KAKAO')
    lbx = Label(wd, text='MILESTONE 3')
    lb1 = Label(wd, text=" INPUT_QUERY      : ")

    text1 = Entry(wd, textvariable=origin, width=50)
    btn = Button(wd, text="Show the result",
                 command=calculate)  # 마우스로 show the result 버튼 클릭 해서 가능하게

    # 위젯 배치 (grid 방식) like 표
    lb0.grid(row=0, column=1)
    lbx.grid(row=1, column=1)
    lb1.grid(row=2, column=0)
    text1.grid(row=2, column=1)
    btn.grid(row=3, column=1)
    wd.bind('<Escape>', exit)  # esc 누르면 종료하게
    # 메인루프
    wd.mainloop()
