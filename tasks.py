from invoke import task

@task
def delete(c):
    c.run("rm -rf {}".format('*.so'))

@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")

@task
def run(c,k=None,n=None,Random=True):
    c.run("python3.8.5 -m invoke build")
    if (Random==True):
        c.run("python3.8.5 main.py 0 0 1")
    else:
        if (k==None or n==None):
            c.run("python3.8.5 main.py 0")
        else:
            c.run("python3.8.5 main.py "+str(k)+" "+str(n)+" 0")