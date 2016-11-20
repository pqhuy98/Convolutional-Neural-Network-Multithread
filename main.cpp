#include <stdio.h>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <math.h>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <mingw.thread.h>
#include <ctime>
using namespace std;
#define fi "input.txt"
#define fo "output.txt"
#define fileopen freopen(fi,"r",stdin);freopen(fo,"w",stdout)
#define FOR(i,l,r) for(int i=(int)(l);i<=(int)(r);i++)
#define FORD(i,l,r) for(int i=(int)(l);i>=(int)(r);i--)
#define xy pair<int,int>
#define int64 long long
#define X first
#define Y second
#define pb push_back
#define init(a,v) memset(a,v,sizeof(a))
#define Sz(s) (int)(s.size())
#define EL printf("\n")
#define digit(x) ('0'<=x&&x<='9')
#define ran(l,r) (rand()%((int)(r)-(int)(l)+1)+(int)(l))
#define forever while (true)

const int OO = (int) 2e9;
const int MOD = (int) 1e9+7;
const long double Pi = 3.1415926535897932384626433832795;
const int N = (int) 2e5+5;

const double EPS = 1e-9;
const int iter = 1000000;
const double eps = 1;
const int DEBUG = 0;
const int CONVRELU = 0;
const int MAXPOOL = 1;
const int SOFTMAX = 2;

const int Ntrain = 500;
const int Ntest = 100;
const char traintestpath[] = "";

typedef vector<int> vi;
typedef vector<vi> vii;
typedef vector<vii> viii;
typedef vector<viii> viiii;
typedef vector<double> vd;
typedef vector<vd> vdd;
typedef vector<vdd> vddd;
typedef vector<vddd> vdddd;
typedef vector<vdddd> vddddd;

void reset(vd &a) {
    fill(a.begin(),a.end(),0);
}
void reset(vdd &a) {
    FOR(i,0,Sz(a)-1) reset(a[i]);
}
void reset(vddd &a) {
    FOR(i,0,Sz(a)-1) reset(a[i]);
}
void reset(vdddd &a) {
    FOR(i,0,Sz(a)-1) reset(a[i]);
}
void reset(vddddd &a) {
    FOR(i,0,Sz(a)-1) reset(a[i]);
}
void resize(vdd &a,int n,int m) {
    a.resize(n);
    FOR(i,0,n-1) a[i].resize(m);
}
void resize(vddd &a,int n,int m,int v) {
    a.resize(n);
    FOR(i,0,n-1) resize(a[i],m,v);
}
void resize(vii &a,int n,int m) {
    a.resize(n);
    FOR(i,0,n-1) a[i].resize(m);
}
void resize(viii &a,int n,int m,int v) {
    a.resize(n);
    FOR(i,0,n-1) resize(a[i],m,v);
}
void resize(vdddd &a,int n,int m,int v,int k) {
    a.resize(n);
    FOR(i,0,n-1) resize(a[i],m,v,k);
}
double ranw(double eps) {
    double res=ran(1,30000)/30000.0*eps;
    return res;
}
void ranw(vd &a,double eps) {
    FOR(i,0,Sz(a)-1) a[i]=ranw(eps);
}
void ranw(vdd &a,double eps) {
    FOR(i,0,Sz(a)-1) ranw(a[i],eps);
}
void ranw(vddd &a,double eps) {
    FOR(i,0,Sz(a)-1) ranw(a[i],eps);
}
void ranw(vdddd &a,double eps) {
    FOR(i,0,Sz(a)-1) ranw(a[i],eps);
}

struct neural_network{
    int nlayer;
    double learnconst,normconst,preL;
    vi layertype,width,height,filter,step,stride,pad0;
    viiii prex,prey;
    vdd b,bbuffer;
    vdddd a,z,e;
    vddddd w,wbuffer;
    /*******************************READ*AND*WRITE*DATA******************************/
    void read(const char *filename) {
        freopen(filename,"r",stdin);
        preL=1e18;
        cin>>nlayer>>learnconst>>normconst;
        layertype.resize(nlayer);
        width.resize(nlayer);height.resize(nlayer);filter.resize(nlayer);
        step.resize(nlayer);stride.resize(nlayer);pad0.resize(nlayer);
        prex.resize(nlayer);prey.resize(nlayer);
        a.resize(nlayer);z.resize(nlayer);e.resize(nlayer);
        b.resize(nlayer);bbuffer.resize(nlayer);
        w.resize(nlayer);wbuffer.resize(nlayer);
        string header;int random;double ew,eb;
        scanf("%i%i%i",&width[0],&height[0],&filter[0]);
        resize(a[0],height[0],width[0],filter[0]);
        FOR(l,1,nlayer-1) {
            cin>>header;
            if (header=="CONVRELU") {
                layertype[l]=CONVRELU;
                scanf("%i%i%i%i",&filter[l],&step[l],&stride[l],&pad0[l]);
            } else if (header=="MAXPOOL") {
                layertype[l]=MAXPOOL;
                scanf("%i%i%i",&step[l],&stride[l],&pad0[l]);
                filter[l]=filter[l-1];
            } else if (header=="SOFTMAX") {
                layertype[l]=SOFTMAX;
                filter[l]=filter[l-1];
                step[l]=1;
                stride[l]=1;
                pad0[l]=0;
            }
            height[l]=(height[l-1]-step[l]+2*pad0[l])/stride[l]+1;
            width[l]=(width[l-1]-step[l]+2*pad0[l])/stride[l]+1;
            resize(a[l],height[l],width[l],filter[l]);
            resize(z[l],height[l],width[l],filter[l]);
            resize(e[l],height[l],width[l],filter[l]);
            if (layertype[l]==CONVRELU) {
                resize(w[l],step[l],step[l],filter[l],filter[l-1]);
                resize(wbuffer[l],step[l],step[l],filter[l],filter[l-1]);
                b[l].resize(filter[l]);
                bbuffer[l].resize(filter[l]);
            } else if (layertype[l]==MAXPOOL) {
                resize(prex[l],width[l],height[l],filter[l]);
                resize(prey[l],width[l],height[l],filter[l]);
            }
            if (layertype[l]==CONVRELU) {
                scanf("%i",&random);
                if (random) {
                    scanf("%lf %lf",&ew,&eb);
                    ranw(w[l],ew);
                    FOR(t,0,filter[l]-1) b[l][t]=eb;
                } else {
                    FOR(i,0,step[l]-1) FOR(j,0,step[l]-1) FOR(t,0,filter[l]-1)
                        FOR(t1,0,filter[l-1]-1) scanf("%lf",&w[l][i][j][t][t1]);
                    FOR(t,0,filter[l]-1) scanf("%lf",&b[l][t]);
                }
            }
        }
        FOR(i,0,3) {
            ta[i]=a,tz[i]=z,te[i]=e;
            tprex[i]=prex;tprey[i]=prey;
            twbuffer[i]=wbuffer;
            tbbuffer[i]=bbuffer;
            reset(twbuffer[i]);
            reset(tbbuffer[i]);
        }
        fclose(stdin);
    }
    void write(const char*filename) {
        ofstream f(filename);
        f.precision(18);
        f<<nlayer<<" "<<learnconst<<" "<<normconst<<endl;
        f<<height[0]<<" "<<width[0]<<" "<<filter[0]<<endl;
        FOR(l,1,nlayer-1) {
            if (layertype[l]==CONVRELU) {
                f<<"CONVRELU"<<endl;
                f<<filter[l]<<" "<<step[l]<<" "<<stride[l]<<" "<<pad0[l]<<endl;
                f<<"0"<<endl;
                FOR(i,0,step[l]-1) {
                    FOR(j,0,step[l]-1) {
                        FOR(t,0,filter[l]-1) {
                            FOR(t1,0,filter[l-1]-1) {
                                f<<w[l][i][j][t][t1]<<" ";
                            }
                            f<<endl;
                        }
                    }
                }
                FOR(t,0,filter[l]-1) f<<b[l][t]<<" ";
                f<<endl;
            } else if (layertype[l]==MAXPOOL) {
                f<<"MAXPOOL"<<endl;
                f<<step[l]<<" "<<stride[l]<<" "<<pad0[l]<<endl;
            } else if (layertype[l]==SOFTMAX) f<<"SOFTMAX"<<endl;
        }
        f.close();
    }
    /*******************************FUNCTION********************************/
    double f(double x) {
        return max(0.0,x);
    }
    double df(double x) {
        if (x<0) return 0; else return 1;
    }
    double L(double x,double y) {
        if (abs(x-y)<EPS) return 0; else
        if (y+EPS>1) return -y*log(x);
        else return -log(1-x);
    }
    double dL(double x,double y) {
        if (abs(x-y)<EPS) return 0;
        return (x-y)/(x-x*x);
    }
    /***************************FEED**FORWARD*****************************/
    /*  a[l-1] ==w[l]==> a[l]  */
    void feedforw(vddd &input) {
        int xp,yp,x,y;
        a[0]=input;
        FOR(l,1,nlayer-2) {
            FOR(i,0,height[l]-1) FOR(j,0,width[l]-1) FOR(t,0,filter[l]-1) {
                if (layertype[l]==CONVRELU) z[l][i][j][t]=b[l][t];
                else prex[l][i][j][t]=prey[l][i][j][t]=-1;
                FOR(i1,0,step[l]-1) FOR(j1,0,step[l]-1) {
                    xp=i*stride[l]-pad0[l]+i1;
                    yp=j*stride[l]-pad0[l]+j1;
                    if (xp<0||xp>=height[l-1]||yp<0||yp>=width[l-1]) continue;
                    if (layertype[l]==0) FOR(t1,0,filter[l-1]-1)
                        z[l][i][j][t]+=a[l-1][xp][yp][t1]*w[l][i1][j1][t][t1];
                    else {
                        x=prex[l][i][j][t];y=prey[l][i][j][t];
                        if (x<0||y<0) prex[l][i][j][t]=x=xp,prey[l][i][j][t]=y=yp;
                        if (a[l-1][xp][yp][t]>a[l-1][x][y][t])
                            prex[l][i][j][t]=xp,prey[l][i][j][t]=yp;
                    }
                }
                if (layertype[l]==0) a[l][i][j][t]=f(z[l][i][j][t]);
                else a[l][i][j][t]=z[l][i][j][t]=a[l-1][prex[l][i][j][t]][prey[l][i][j][t]][t];
            }
        }
        int l=nlayer-1; double sum=0;
        //FOR(t,0,filter[l]-1) cout<<"Prob "<<t<<" : "<<a[l][0][0][t]<<endl;
        FOR(t,0,filter[l]-1) a[l][0][0][t]=exp(a[l-1][0][0][t]),sum+=a[l][0][0][t];
        FOR(t,0,filter[l]-1) a[l][0][0][t]/=sum;
    }
    /*************************BACK**PROB*********************************/
    void backprob(vd &output) {
        int xp,yp,x=nlayer-1,y;
        reset(e);
        FOR(i,0,filter[x]-1) e[x-1][0][0][i]=a[x][0][0][i]-output[i];
        FORD(l,nlayer-2,1) FOR(i,0,height[l]-1) FOR(j,0,width[l]-1) FOR(t,0,filter[l]-1) {
            if (layertype[l]==0) e[l][i][j][t]*=df(z[l][i][j][t]);
            if (layertype[l]==0) {
                FOR(i1,0,step[l]-1) FOR(j1,0,step[l]-1) {
                    xp=i*stride[l]-pad0[l]+i1;
                    yp=j*stride[l]-pad0[l]+j1;
                    if (xp<0||xp>=height[l-1]||yp<0||yp>=width[l-1]) continue;
                    FOR(t1,0,filter[l-1]-1) {
                        if (l-1) e[l-1][xp][yp][t1]+=e[l][i][j][t]*w[l][i1][j1][t][t1];
                        wbuffer[l][i1][j1][t][t1]+=e[l][i][j][t]*a[l-1][xp][yp][t1];
                    }
                }
                bbuffer[l][t]+=e[l][i][j][t];
            } else {
                x=prex[l][i][j][t];y=prey[l][i][j][t];
                if (l-1) e[l-1][x][y][t]=e[l][i][j][t];
            }
        }
    }
    double update() {
        double res=0;
        FOR(l,1,nlayer-1) if (layertype[l]==0) {
            FOR(i,0,step[l]-1) FOR(j,0,step[l]-1) FOR(t,0,filter[l]-1) FOR(t1,0,filter[l-1]-1) {
                res+=abs(wbuffer[l][i][j][t][t1]+normconst*w[l][i][j][t][t1]),
                w[l][i][j][t][t1]-=learnconst*(wbuffer[l][i][j][t][t1]+normconst*w[l][i][j][t][t1]);
                wbuffer[l][i][j][t][t1]=0;
            }
            FOR(t,0,filter[l]-1)
                res+=abs(bbuffer[l][t]+normconst*b[l][t]),
                b[l][t]-=learnconst*(bbuffer[l][t]+normconst*b[l][t]),
                bbuffer[l][t]=0;;
        }
        cout<<"D = "<<res<<" -- ";
        return res;
    }
    /**********************LOSS**FUNCTION****************************/
    double L(vd &output) {
        double res=0;
        FOR(i,0,Sz(output)-1)
                res+=L(a[nlayer-1][0][0][i],output[i]);
        return res;
    }
    /**********************LEARN*************************************/
    double learn(vdddd &input,vdd &output,int upd) {
        double res=0;
        if (upd) {
            reset(wbuffer);
            reset(bbuffer);
        }
        FOR(t,0,Sz(input)-1) {
            feedforw(input[t]);
            backprob(output[t]);
            res+=L(output[t]);
            FOR(i,0,Sz(input[t])-1)
                reverse(input[t][i].begin(),input[t][i].end());
            feedforw(input[t]);
            backprob(output[t]);
            res+=L(output[t]);
        }
        if (upd) {
            update();
            //test(input[0],output[0]);
        }
        res/=Sz(input);
        if (upd&&res>preL) cout<<"LEARNING RATE IS TOO LARGE !"<<endl,learnconst*=0.9;
        if (upd) preL=res;
        return res;
    }
    /*********************OUTPUT************************************/
    vd output(vddd &input) {
        feedforw(input);
        return a[nlayer-1][0][0];
    }
    int output1(vddd &input) {
        vd o=output(input);
        int res=0;
        FOR(i,0,9) if (o[res]<o[i]) res=i;
        return res;
    }
    vi output1(vdddd &input) {
        vi res;res.resize(Sz(input));
        FOR(i,0,Sz(input)-1) {
            res[i]=output1(input[i]);
            cout<<"OUTPUT "<<i<<endl;
        }return res;
    }

    void test(vddd &input,vd &output) {
        double L1,L2,dL,wb;
        cout<<"TEST============================"<<endl;
        FOR(l,1,nlayer-1) if (layertype[l]==0) {
            cout<<"WEIGHT"<<endl;
            FOR(i,0,step[l]-1) FOR(j,0,step[l]-1) FOR(t,0,filter[l]-1) FOR(t1,0,filter[l-1]-1) {
                w[l][i][j][t][t1]-=EPS;
                feedforw(input);
                L1=L(output);
                w[l][i][j][t][t1]+=2*EPS;
                feedforw(input);
                L2=L(output);
                dL=(L2-L1)/(2*EPS);
                w[l][i][j][t][t1]-=EPS;
                feedforw(input);
                wbuffer[l][i][j][t][t1]=0;
                backprob(output);
                wb=wbuffer[l][i][j][t][t1];
                dL-=wbuffer[l][i][j][t][t1];
                cout<<dL<<endl;
            }
            cout<<"BIAS"<<endl;
            FOR(t,0,filter[l]-1) {
                b[l][t]-=EPS;
                feedforw(input);
                L1=L(output);
                b[l][t]+=2*EPS;
                feedforw(input);
                L2=L(output);
                dL=(L2-L1)/(2*EPS);
                b[l][t]-=EPS;
                feedforw(input);
                bbuffer[l][t]=0;
                backprob(output);
                wb=bbuffer[l][t];
                dL-=wb;
                cout<<dL<<endl;
            }
        }
    }
/******************************** MULTI - THREAD **********************************/
    vdddd ta[4],tz[4],te[4];
    viiii tprex[4],tprey[4];
    vdd tbbuffer[4];
    vddddd twbuffer[4];

    double tL(int id,vd &output) {
        double res=0;
        FOR(i,0,Sz(output)-1)
                res+=L(ta[id][nlayer-1][0][0][i],output[i]);
        return res;
    }
    void MTff(int id,vddd &input) {
        int xp,yp,x,y;
        ta[id][0]=input;
        FOR(l,1,nlayer-2) {
            FOR(i,0,height[l]-1) FOR(j,0,width[l]-1) FOR(t,0,filter[l]-1) {
                if (layertype[l]==CONVRELU) tz[id][l][i][j][t]=b[l][t];
                else tprex[id][l][i][j][t]=tprey[id][l][i][j][t]=-1;
                FOR(i1,0,step[l]-1) FOR(j1,0,step[l]-1) {
                    xp=i*stride[l]-pad0[l]+i1;
                    yp=j*stride[l]-pad0[l]+j1;
                    if (xp<0||xp>=height[l-1]||yp<0||yp>=width[l-1]) continue;
                    if (layertype[l]==0) FOR(t1,0,filter[l-1]-1)
                        tz[id][l][i][j][t]+=ta[id][l-1][xp][yp][t1]*w[l][i1][j1][t][t1];
                    else {
                        x=tprex[id][l][i][j][t];y=tprey[id][l][i][j][t];
                        if (x<0||y<0) tprex[id][l][i][j][t]=x=xp,tprey[id][l][i][j][t]=y=yp;
                        if (ta[id][l-1][xp][yp][t]>ta[id][l-1][x][y][t])
                            tprex[id][l][i][j][t]=xp,tprey[id][l][i][j][t]=yp;
                    }
                }
                if (layertype[l]==0) ta[id][l][i][j][t]=f(tz[id][l][i][j][t]);
                else ta[id][l][i][j][t]=tz[id][l][i][j][t]=ta[id][l-1][tprex[id][l][i][j][t]][tprey[id][l][i][j][t]][t];
            }
        }
        int l=nlayer-1; double sum=0;
        //FOR(t,0,filter[l]-1) cout<<"Prob "<<t<<" : "<<a[l][0][0][t]<<endl;
        FOR(t,0,filter[l]-1) ta[id][l][0][0][t]=exp(ta[id][l-1][0][0][t]),sum+=ta[id][l][0][0][t];
        FOR(t,0,filter[l]-1) ta[id][l][0][0][t]/=sum;
    }

    void MTbp(int id,vd &output) {
        int xp,yp,x=nlayer-1,y;
        reset(te[id]);
        FOR(i,0,filter[x]-1) te[id][x-1][0][0][i]=ta[id][x][0][0][i]-output[i];
        FORD(l,nlayer-2,1) FOR(i,0,height[l]-1) FOR(j,0,width[l]-1) FOR(t,0,filter[l]-1) {
            if (layertype[l]==0) te[id][l][i][j][t]*=df(tz[id][l][i][j][t]);
            if (layertype[l]==0) {
                FOR(i1,0,step[l]-1) FOR(j1,0,step[l]-1) {
                    xp=i*stride[l]-pad0[l]+i1;
                    yp=j*stride[l]-pad0[l]+j1;
                    if (xp<0||xp>=height[l-1]||yp<0||yp>=width[l-1]) continue;
                    FOR(t1,0,filter[l-1]-1) {
                        if (l-1) te[id][l-1][xp][yp][t1]+=te[id][l][i][j][t]*w[l][i1][j1][t][t1];
                        twbuffer[id][l][i1][j1][t][t1]+=te[id][l][i][j][t]*ta[id][l-1][xp][yp][t1];
                    }
                }
                tbbuffer[id][l][t]+=te[id][l][i][j][t];
            } else {
                x=tprex[id][l][i][j][t];y=tprey[id][l][i][j][t];
                if (l-1) te[id][l-1][x][y][t]=te[id][l][i][j][t];
            }
        }
    }
    double MTupd(int nThread) {
        double res=0,sumt;
        FOR(l,1,nlayer-1) if (layertype[l]==CONVRELU) {
            FOR(i,0,step[l]-1) FOR(j,0,step[l]-1) FOR(t,0,filter[l]-1) FOR(t1,0,filter[l-1]-1) {
                sumt=0;
                FOR(id,0,nThread-1) sumt+=twbuffer[id][l][i][j][t][t1],twbuffer[id][l][i][j][t][t1]=0;
                res+=abs(sumt+normconst*w[l][i][j][t][t1]),
                w[l][i][j][t][t1]-=learnconst*(sumt+normconst*w[l][i][j][t][t1]);
            }
            FOR(t,0,filter[l]-1) {
                sumt=0;
                FOR(id,0,nThread-1) sumt+=tbbuffer[id][l][t],tbbuffer[id][l][t]=0;
                res+=abs(sumt+normconst*b[l][t]),
                b[l][t]-=learnconst*(sumt+normconst*b[l][t]);
            }
        }
        cout<<"D = "<<res<<" -- ";
        return res;
    }
    double rs[4];
    void ffbp(int id,vdddd &input,vdd &output,int start,int cnt) {
        FOR(t,start,start+cnt-1) {
            MTff(id,input[t]);
            MTbp(id,output[t]);
            rs[id]+=tL(id,output[t]);
            FOR(i,0,Sz(input[t])-1)
                reverse(input[t][i].begin(),input[t][i].end());
            MTff(id,input[t]);
            MTbp(id,output[t]);
            rs[id]+=tL(id,output[t]);
        }
        //cout<<"Thread "<<id<<" done."<<endl;
    }
    double MTlearn(int nThread,vdddd &input,vdd &output,int upd) {
        if (upd) FOR(i,0,nThread-1) {
            reset(tbbuffer[i]);
            reset(twbuffer[i]);
        }
        //cout<<"Learning Const : "<<learnconst<<endl;
        double res=0;
        FOR(i,0,nThread-1) rs[i]=0;
        int len=Sz(input)/nThread,rem=Sz(input)%nThread;
        thread thr[4];
        FOR(i,0,nThread-2)
            thr[i]=thread(&neural_network::ffbp,this,i,input,output,i*len,len);
        thr[nThread-1]=thread(&neural_network::ffbp,this,nThread-1,input,output,(nThread-1)*len,len+rem);
        FOR(i,0,nThread-1) thr[i].join();
        if (upd) {
            MTupd(nThread);
        }
        FOR(i,0,nThread-1) res+=rs[i];
        res/=Sz(input);
        if (upd&&res>preL) cout<<"LEARNING RATE IS TOO LARGE !"<<endl,learnconst*=0.9;
        if (upd) preL=res;
        return res;
    }
    double MTlearn2(int id,vdddd &input,vdd &output) {
        double res=0;
        rs[id]=0;
        ffbp(id,input,output,0,Sz(input));
        res=rs[id]/Sz(input);
        return res;
    }
    void MTtest(int nt,vdddd &input,vdd &output) {
        reset(wbuffer);
        reset(bbuffer);
        FOR(i,0,nt-1) {
            reset(twbuffer[i]);
            reset(tbbuffer[i]);
        }
        cout<<"Normal learning"<<endl;
        learn(input,output,0);
        cout<<"Thread learning"<<endl;
        MTlearn(nt,input,output,0);
        FOR(l,1,nlayer-1) if (layertype[l]==0) {
            FOR(i,0,step[l]-1) FOR(j,0,step[l]-1) FOR(t,0,filter[l]-1) FOR(t1,0,filter[l-1]-1)
                FOR(id,0,nt-1) wbuffer[l][i][j][t][t1]-=twbuffer[id][l][i][j][t][t1];
            FOR(t,0,filter[l]-1) FOR(id,0,nt-1) bbuffer[l][t]-=tbbuffer[id][l][t];;
        }
        FOR(l,1,nlayer-1) if (layertype[l]==0) {
            FOR(i,0,step[l]-1) FOR(j,0,step[l]-1) FOR(t,0,filter[l]-1) FOR(t1,0,filter[l-1]-1)
                cout<<"Weight "<<wbuffer[l][i][j][t][t1]<<endl;
            FOR(t,0,filter[l]-1) cout<<"Bias "<<bbuffer[l][t]<<endl;
        }
    }
    double MTtest2(int nt) {
        double s,mx=0;
        FOR(l,1,nlayer-1) if (layertype[l]==0) {
            FOR(i,0,step[l]-1) FOR(j,0,step[l]-1) FOR(t,0,filter[l]-1) FOR(t1,0,filter[l-1]-1) {
                s=0;
                FOR(id,0,nt-1) s+=twbuffer[id][l][i][j][t][t1];
                mx=max(abs(wbuffer[l][i][j][t][t1]-s),mx);
            }
            FOR(t,0,filter[l]-1) {
                s=0;
                FOR(id,0,nt-1) s+=tbbuffer[id][l][t];;
                mx=max(abs(bbuffer[l][t]-s),mx);
            }
        }
        cout<<"dL/dw diff = "<<mx<<endl;
        return mx;
    }
} NN;


vdddd input;vdd output;

/*****************************************TRAINING**AND**TESTING****************************/
int maxscore=0;

void readset(const char *filename,vdddd &input,vdd &output) {
    //cout<<filename<<endl;
    unsigned char data[5000];
    FILE *f=fopen(filename,"rb");
    int setsize=100,h=32,w=32,F=3,osz=10;
    resize(input,setsize,h,w,F);
    resize(output,setsize,osz);
    FOR(t,0,Sz(input)-1) {
        fread(data,sizeof(unsigned char),3073,f);
        unsigned char label=data[0],x;int pos=0;
        reset(output[t]);output[t][label]=1;
        FOR(i,0,h-1) FOR(j,0,w-1) {
            x=data[++pos];
            input[t][i][j][0]=1.0*x/255;
        }
        FOR(i,0,h-1) FOR(j,0,w-1) {
            x=data[++pos];
            input[t][i][j][1]=1.0*x/255;
        }
        FOR(i,0,h-1) FOR(j,0,w-1) {
            x=data[++pos];
            input[t][i][j][2]=1.0*x/255;
        }
    }
    fclose(f);
}
void itoa(char *s,int x) {
    int l=0;
    while (x) s[l++]=x%10+'0',x/=10;
    reverse(s,s+l);
    s[l]=0;
}
void readset(const char *name,int pos,vdddd &input,vdd &output) {
    char path[50],id[50];
    path[0]=id[0]=0;
    itoa(id,pos);
    strcat(path,name);
    strcat(path,id);
    strcat(path,".bin");
    readset(path,input,output);
}
double train(int pos) {
    NN.preL=1e18;double Loss,res=0;
    cout<<"Training on "<<pos<<"..."<<endl;
    readset("CIFAR10train",pos,input,output);
    FOR(_,1,3) {
        //Loss=NN.MTlearn(3,input,output,1);
        Loss=NN.learn(input,output,1);
        cout<<"Loss = "<<Loss<<endl;
        res+=Loss;
    }
    res/=3;
    return res;
}

double tmp[4];
vi batchid;
vdddd tinput[4];
vdd toutput[4];

void MTtrainmulti(int id,int start,int len) {
    FOR(i,start,start+len-1) {
        readset("CIFAR10train",batchid[i],tinput[id],toutput[id]);
        tmp[id]+=NN.MTlearn2(id,tinput[id],toutput[id]);
        cout<<batchid[i]<<endl;
    }
}

double trainmulti(int core,int cnt,int times) {
    double res,sumres=0,preL=1e18;
    bool taken;
    cout<<"MULTI TRAIN on ";
    FOR(i,0,core-1) {
        reset(NN.twbuffer[i]);
        reset(NN.tbbuffer[i]);
    }
    cnt*=core;
    batchid.clear();
    FOR(i,0,cnt-1) {
        while (true) {
            taken=false;
            batchid.pb(ran(1,Ntrain));
            FOR(j,0,i-1) if (batchid[i]==batchid[j]) {taken=true;break;}
            if (!taken) break; else batchid.pop_back();
        }
        cout<<batchid.back()<<" ";
    }
    EL;
    FOR(t,1,times) {
        thread thr[4];
        cout<<"-----------Step "<<t<<"---------"<<endl;
        res=0;
        FOR(i,0,core-1) tmp[i]=0;
        FOR(i,0,core-1)
            thr[i]=thread(MTtrainmulti,i,i*(cnt/core),cnt/core);
        FOR(i,0,core-1) thr[i].join();
        FOR(i,0,core-1) res+=tmp[i];
        res/=cnt;
        if (res>preL) cout<<"TRAINING RATE IS TOO LARGE !"<<endl;
        preL=res;
        NN.learnconst/=cnt;
        NN.MTupd(core);
        NN.learnconst*=cnt;
        cout<<"Average Loss = "<<res<<endl;
        sumres+=res;

    }
    return sumres;
}
double trainmulti2(int core,int cnt,int times) {
    double res,sumres=0,preL=1e18;
    bool taken;
    cout<<"MULTI TRAIN on ";
    FOR(i,0,core-1) {
        reset(NN.twbuffer[i]);
        reset(NN.tbbuffer[i]);
    }
    cnt*=core;
    batchid.clear();
    FOR(i,0,cnt-1) {
        while (true) {
            taken=false;
            batchid.pb(ran(1,Ntrain));
            FOR(j,0,i-1) if (batchid[i]==batchid[j]) {taken=true;break;}
            if (!taken) break; else batchid.pop_back(),i--;
        }
        cout<<batchid.back()<<" ";
    }
    EL;
    FOR(t,1,times) {
        cout<<"-----------Step "<<t<<"---------"<<endl;
        res=0;
        FOR(i,0,Sz(batchid)-1) {
            readset("CIFAR10train",batchid[i],input,output);
            res+=NN.MTlearn(core,input,output,0);
            cout<<batchid[i]<<endl;
        }
        res/=cnt;
        if (res>preL) cout<<"TRAINING RATE IS TOO LARGE !"<<endl;
        preL=res;
        NN.learnconst/=cnt;
        NN.MTupd(core);
        NN.learnconst*=cnt;
        cout<<"Average Loss = "<<res<<endl;
        sumres+=res;

    }
    return sumres;
}
int test(int pos) {
    int result=0;vd out;
    readset("CIFAR10test",pos,input,output);
    FOR(t,0,Sz(input)-1) {
        out=NN.output(input[t]);
        int res=0,ans=0;
        FOR(i,0,9) {
            if (output[t][res]<output[t][i]) res=i;
            if (out[ans]<out[i]) ans=i;
        }
        if (res==ans) result++;
    }
    return result;
}
int TESTINGSESSION() {
    int score=0;
    cout<<"TESTING SESSION :"<<endl;
    //cout<<score<<" ";
    FOR(_,1,Ntest) {
        score+=test(_);
        //cout<<score<<" ";
        cout<<_<<" ";
        if (_%10==0) EL;
    }
    cout<<score<<"/"<<Ntest*Sz(input)<<endl;
    if (score>=maxscore) {
        NN.write("NN_best.txt");
        ofstream f2("score.txt");
        f2<<score;
        f2.close();
        maxscore=score;
        cout<<"NEW SCORE SAVED !"<<endl;
    }
    return score;
}
void TRAININGSESSION(int iter) {
    double sumL=0,cnt=0;
    FOR(i,1,iter) {
        cout<<"====================ITERATION "<<i<<"========================="<<endl;
        clock_t pt=clock();
        sumL+=trainmulti(4,4,1);
        cout<<"Running time : "<<double(clock()-pt)<<endl;
        //sumL+=train(ran(1,Ntrain));
        cnt+=2;
        if (i%10==0) {
            cout<<"Saving data..."<<endl;
            NN.write("NN_data.txt");
            cout<<"Saved !"<<endl;
            cout<<"Max score : "<<maxscore<<endl;
            cout<<"Average Loss : "<<sumL/cnt<<endl;
            if (i%30==0) sumL=0,cnt=0;
        }
        if (i%50==0) TESTINGSESSION();
        EL;
    }
    NN.write("NN_data.txt");
}
/*************************************KAGGLE**INPUT**OUTPUT***************************/
void readCSV(vdddd &input) {
    freopen("test.csv","r",stdin);
    string header;
    cin>>header;
    int x;char tmp;
    resize(input,28000,28,28,1);
    FOR(t,0,Sz(input)-1) {
        cout<<"INPUT "<<t<<endl;
        FOR(i,0,27) FOR(j,0,27) {
            scanf("%i",&x);if (i!=27||j!=27) scanf("%c",&tmp);
            input[t][i][j][0]=1.0*x/255;
        }
    }
}

void solveKAGGLE() {
    cout<<"READING..."<<endl;
    readCSV(input);
    cout<<"DONE"<<endl;
    cout<<"CALCULATING..."<<endl;
    vi ans=NN.output1(input);
    cout<<"DONE"<<endl;
    cout<<"WRITING"<<endl;
    FILE *f=fopen("kaggle01.csv","w");
    fprintf(f,"ImageId,Label\n");
    FOR(i,0,Sz(ans)-1) fprintf(f,"%i,%i\n",i,ans[i]);
    fclose(f);
    cout<<"DONE";
}
/***************************************************************************************/
int main() {
    srand(time(NULL));
    ifstream f1("score.txt");
    f1>>maxscore;
    f1.close();
    NN.read("NN_data.txt");

    TRAININGSESSION (iter);

    NN.write("NN_data.txt");
    return 0;
}
