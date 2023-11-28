#include <bits/stdc++.h>
//#include <thread>
#include <string>
using namespace std;

namespace DataUtils
{
    const double PI=3.1415926;

    vector<string> split(const string& input, const string& reg)
    {
        regex re(reg);
        sregex_token_iterator first{input.begin(), input.end(), re, -1}, last;
        return {first, last};
    }

    string trim(const string &s)
    {
        int i=0;
        while (s[i]==' ' || s[i]=='\t' || s[i]=='\n') ++i;
        int j=s.size()-1;
        while (s[j]==' ' || s[i]=='\t' || s[i]=='\n') --j;
        return s.substr(i,j+1);
    }

    double stoi(const string &str)
    {
        string s=trim(str);
        int flags=0;int ans =0;int i=0;
        if (s[0]=='-') flags=1, ++i;
        while (s[i] && isdigit(s[i])) ans = ans*10 + s[i++]-'0';
        return flags?-ans:ans;
    }

    double stof(const string &str)
    {
        /*
            for double string , e.g. -1.023
            do not support the format 1e2
        */
        string s=trim(str);
        double ans =0;int flags=0;int i=0;
        if (s[0]=='-') flags=1, ++i;
        int pos=s.find(".");
        ans=(double)(stoi(s.substr(i,pos)));
        double tmp=1.0;
        for (i=pos+1;s[i] && isdigit(s[i]);++i) ans=ans+tmp/10*(s[i]-'0'), tmp/=10;
        return flags?-ans:ans;
    }

    bool getDouble(const string line, int &pos, double &ans)
    {
        /*
        get a double from a string
        @line: a string that need to be analysis
        @pos: the start pos of the string that need to be analysis
        @ans: the ans double
        */
        int n = line.length();
        while (pos<n && line[pos]!='-' && line[pos]!='.' && !isdigit(line[pos])) ++pos;
        if (pos==n) return 0;
        int flags=1;
        if (pos<n && line[pos]=='-'){ flags=0; ++pos;}
        if (pos==n) return 0;
        ans = 0;
        while (pos<n && isdigit(line[pos])) ans=ans*10+line[pos]-'0', ++pos;
        double tmp=1;
        if (pos<n && line[pos]=='.') ++pos;
        while(pos<n && isdigit(line[pos])) ans+=(line[pos]-'0')*tmp/10, tmp/=10, ++pos;
        if (!flags) ans=-ans;
        return 1;
    }

    bool getInt(const string line, int &pos, int &ans)
    {
        ans=0;
        int n=line.length();
        while (pos<n && line[pos]!='-' && !isdigit(line[pos])) ++pos;
        if (pos==n) return 0;
        int flags=1;
        if (line[pos]=='-') {flags=0;++pos;}
        if (pos==n) return 0;
        while (pos<n && isdigit(line[pos])) ans=ans*10+line[pos++]-'0';
        if (!flags) ans=-ans;
        return 1;
    }

    double rad(double d) {return d*PI/180.0;}

    double getDistance(double lat1, double lng1, double lat2, double lng2)
    {
        /*
            cal the distance in meters between two points in the world representing as (lat1, lng1, lat2, lng2)
        */
        double radLat1=rad(lat1);
        double radLat2=rad(lat2);
        double a = radLat1-radLat2;
        double b = rad(lng1)-rad(lng2);
        double s = 2 * asin(sqrt(pow(sin(a/2), 2) + cos(radLat1)*cos(radLat2)*pow(sin(b/2),2)));
        s = s * 6378.137;
        return round(s*10000) / 10000;
    }

    struct GeoPoint
    {
        double lat, lon;
        GeoPoint(){lat=0, lon=0;}
        GeoPoint(double lon, double lat):lat(lat), lon(lon){}
        friend ostream& operator<<(ostream& out, GeoPoint& o) {out<<"lon: "<<o.lon<<"\tlat: "<<o.lat<<endl; return out;}
    };

    struct GeoArea
    {
        GeoPoint gp1,gp2;
        int idx;
        GeoArea(){idx=-1;}
        GeoArea(GeoPoint left_buttom, GeoPoint right_top, int idx=-1):gp1(left_buttom),gp2(right_top),idx(idx){}
        GeoArea(double min_lon, double max_lon, double min_lat, double max_lat, int idx=-1):idx(idx)
        {
            gp1=GeoPoint(min_lon, min_lat); gp2=GeoPoint(max_lon, max_lat);
        }
        bool isInArea(GeoPoint gp)
        {
            return gp.lon>=gp1.lon && gp.lon<gp2.lon && gp.lat>=gp1.lat && gp.lat<gp2.lat;
        }
        bool isInArea(const double lon, const double lat)
        {
            return lon>=gp1.lon && lon<gp2.lon && lat>=gp1.lat && lat<gp2.lat;
        }
        friend ostream& operator<<(ostream& out, const GeoArea &o) {out<<"[Area] "<<o.gp1.lon<<"\t"<<o.gp1.lat<<"\t"<<o.gp2.lon<<"\t"<<o.gp2.lat<<endl;return out;}
    };

    struct Datetime
    {
        int year, month, day, hh, mm, ss;
        Datetime(){} //to do initialize
        Datetime(int year, int month ,int day, int hh=0, int mm=0, int ss=0):year(year),month(month),day(day),hh(hh),mm(mm),ss(ss){} // to do analysis the range of every variable
        Datetime(string s, string format="%Y-%m-%d %H:%M:%s")
        {
            //format = "%Y-%m-%d %H:%M:%s"
            if (format!="%Y-%m-%d %H:%M:%s")
            {
                cout<<"do not support this format!"<<endl;
            }
            const char *p=s.c_str();
            sscanf(p,"%d-%d-%d %d:%d:%d", &year, &month, &day, &hh, &mm, &ss);
        }
        ostream& operator<<(ostream& out) {char s[105];sprintf(s,"%04d%02d%02d %02d:%02d:%02d\0",year,month,day,hh,mm,ss);string tmp(s);out<<"[Datetime] "<<tmp<<endl;return out;}
        Datetime operator-(const Datetime& o)
        {
            Datetime ans;
            ans.ss = ss - o.ss;
            if (ans.ss<0) mm-=1, ans.ss+=60;
            ans.mm = mm - o.mm;
            if (ans.mm<0) hh-=1, ans.mm+=60;
            ans.hh = hh - o.hh;
            if (ans.hh<0) day-=1, ans.hh+=24;
            //simplify implement without the day , month, year.
            return ans;
        }
        int get_total_seconds() {return hh*3600+mm*60+ss;}
    };
};


namespace ML
{
    struct Cluster
    {
        set<int> s;
        Cluster(){}
        explicit Cluster(initializer_list<int> l)
        {
            s.clear();s.insert(l);
        }

        friend ostream& operator<<(ostream& out, Cluster o)
        {
            for (auto i:o.s) out<<i<<" ";
            return out;
        }
    };

    double calClusterCompleteDis(const Cluster &a, const Cluster &b, const vector<vector<double>> &sim)
    {
        double ans=0.0;
        for (int i:a.s)
        {
            for (int j:b.s)
            {
                ans=max(ans, sim[i][j]);
            }
        }
        return ans;
    }

    int findMinCluster(list<Cluster> &clusters, const vector<vector<double>> &sim, list<Cluster>::iterator &x, list<Cluster>::iterator &y, double &dis)
    {
        int n=clusters.size();
        if (n<=1) return 0;
        dis=2.0;
        for (auto i=clusters.begin();i!=clusters.end();++i)
        {
            auto j=i;
            ++j;
            for (;j!=clusters.end();++j)
            {
                double tmp = calClusterCompleteDis(*i,*j,sim);
                if (tmp<dis)
                {
                    dis=tmp;x=i;y=j;
                }
            }
        }
        return 1;
    }

    vector<int> HierarchicalCluster(const vector<vector<double>> &sim, double eta)
    {
        /*
            Hierarchical Cluster    [CIKM 2017] Hao WU; A fast trajectory outlier detection approach via driving behavior modeling; Experiment
            1. sim: the distance matrix between every pair
            2. eta: the max distance threshold if the distance between two points exceed the threshold the hierarchical cluster process will be stopped.

            In every steps, find the nearest two point and merge them.
            This program will stop according to eta as we claimed above.
        */

        cout<<"start ML::HierarchicalCluster"<<endl;
        int n=sim.size();
        cout<<"n: "<<n<<endl;
        list<Cluster> clusters;
        for (int i=0;i<n;++i)
        {
            clusters.push_back(Cluster({i}));
        }
        // the point to the nearest pair
        list<Cluster>::iterator cand1,cand2;
        double dis;
        clock_t start_time = clock();
        int rounds=0;
        while (1)
        {
            if (clusters.size()<=1) break;
            bool flags=findMinCluster(clusters, sim, cand1, cand2, dis);
            if (!flags) break;
            if (dis>eta) break;
            // merge to cluster
            Cluster new_cluster;
            new_cluster.s.insert((*cand1).s.begin(), (*cand1).s.end());
            new_cluster.s.insert((*cand2).s.begin(), (*cand2).s.end());
            clusters.push_back(new_cluster);
            clusters.erase(cand2);
            clusters.erase(cand1);
            ++rounds;
            if (rounds%400==0) {cout<<"run one round with size: "<<clusters.size()<<" cost time: "<<(clock()-start_time)/CLOCKS_PER_SEC<<" seconds."<<endl;start_time=clock();}
        }
        vector<int> res(n);
        int labels=0;
        for (auto i:clusters)
        {
            ++labels;
            // cout<<labels<<" : "<<i<<endl;
            for (auto j:i.s) res[j]=labels;
        }
        return res;
    }


};


namespace Porto
{
    /*
        Porto data: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
        The idea is the same as MTA dataset.
    */

    const double MIN_LON=-9, MAX_LON=-8, MIN_LAT=41, MAX_LAT=42;
    const int NUM_LONS=840, NUM_LATS=1100;
    const double DELTA_LON = (MAX_LON-MIN_LON) / NUM_LONS;
    const double DELTA_LAT = (MAX_LAT-MIN_LAT) / NUM_LATS;
    DataUtils::GeoPoint LEFT_BOTTOM(MIN_LON, MIN_LAT), RIGHT_TOP(MAX_LON, MAX_LAT);
    DataUtils::GeoArea AREA(LEFT_BOTTOM, RIGHT_TOP);
    const char *INPUT_DIR="data/Porto/";
    const char *OUTPUT_DIR="output data/Porto/";
    const char *train_set="output data/Porto/train_set.txt";
    const char *test_set = "output data/Porto/test_set.txt";


    void info()
    {
        cout<<"Porto datasets basic information**************************************************************"<<endl;
        cout<<"[Area Range]\t"<<"MIN_LON: "<<MIN_LON<<"\tMAX_LON: "<<MAX_LON<<"\tMIN_LAT: "<<MIN_LAT<<"\tMAX_LAT"<<MAX_LAT<<endl;
        cout<<"[Area Split Nums]\t"<<"NUM_LONS: "<<NUM_LONS<<"\tNUM_LATS: "<<NUM_LATS<<endl;
        cout<<"[Area split Delta]\t"<<"DELTA_LONS: "<<DELTA_LON<<"\tDELTA_LAT: "<<DELTA_LAT<<endl;
        cout<<"**********************************************************************************************"<<endl;
    }

    int getAreaID(double lon, double lat)
    {
        if (!AREA.isInArea(lon, lat)) return -1;
        int i = int((lat-MIN_LAT) / DELTA_LAT);
        int j = int((lon-MIN_LON) / DELTA_LON);
        assert(0<=j && j<NUM_LONS);
        assert(0<=i && i<NUM_LATS);
        return i * NUM_LONS + j;
    }

    bool isInGrid(int i, int j)
    {
        return i>=0 && i < NUM_LATS && j>=0 && j<NUM_LONS;
    }

    set<int> getAreaNeighbors(int id, int steps=1) {
        set<int> ans;
        // ans.insert(id);
        int i=id/NUM_LONS;
        int j=id%NUM_LONS;
        for (int x=i-steps;x<=i+steps;++x) {
            for (int y=j-steps;y<=j+steps;++y)
            {
                if (isInGrid(x, y)) ans.insert(x*NUM_LONS+y);
            }
        }
        return ans;
    }

    bool isConnect(int i,int j)
    {
        set<int> ans =getAreaNeighbors(i);
        if (ans.find(j)!=ans.end()) return true;
        return false;
    }

    DataUtils::GeoArea getArea(int id)
    {
        int i = id/NUM_LONS;
        int j = id%NUM_LONS;
        return DataUtils::GeoArea(MIN_LON+j*DELTA_LON, MIN_LON+(j+1)*DELTA_LON, MIN_LAT+i*DELTA_LAT, MIN_LAT+(i+1)*DELTA_LAT);
    }

    int splitLines(const int slines,const int elines, const string infilename, const string outfilename)
    {
        cout<<"start thread"<<"\tslines: "<<slines<<"\telines: "<<elines<<"\twrite to file: "<<outfilename<<endl;
        ifstream fin(infilename.c_str());
        ofstream fout(outfilename.c_str());
        int line_num = 0;
        string line;
        while (line_num<slines) getline(fin,line), ++line_num;
        vector<int> ans;ans.clear();
        double pre_lon, pre_lat; //前一点的位置
        double lat, lon;         //当前点的位置
        int idx;
        long long etime, pretime;
        while (line_num<elines && getline(fin, line))
        {
            if (line_num++==0) continue;
            int tpos=0;for (int i=0;i<4;++i) tpos=line.find("\",\"", tpos), tpos+=3;
            sscanf(line.substr(tpos).c_str(),"%d\",\"%lld\"%*s", &idx, &etime);
            // cout<<"idx: "<<idx<<"\tetime: "<<etime<<endl;
            int pos = line.rfind("\",\"");
            while (DataUtils::getDouble(line, pos, lon) && DataUtils::getDouble(line, pos, lat))
            {
                // 判断非法情况 time spatial
                bool flags=0;
                double dis = DataUtils::getDistance(lat, lon, pre_lat, pre_lon);
                pre_lat=lat; pre_lon=lon;
                if (dis>1) flags=1;
                int ids = getAreaID(lon, lat);
                if (ids==-1) flags=1;
                DataUtils::GeoArea area= getArea(ids);
                // cout<<"lon: "<<lon<<"\tlat: "<<lat<<"\tid: "<<ids<<"\tArea: "<<area<<"\tin: "<<area.isInArea(lon,lat)<<endl;
                if (flags && ans.size())
                {
                    //cout<<"out happened!"<<endl;
                    fout<<idx;for (auto i:ans) fout<<" "<<i;fout<<endl;ans.clear();continue;
                }
                if (ids!= -1 && (ans.size()==0 || ans.back() != ids)) ans.push_back(ids);
            }
            if (ans.size())
            {
                fout<<idx;for (auto i:ans) fout<<" "<<i;fout<<endl;ans.clear();
            }
            if ((line_num-slines)%5000==0) cout<<"process..."<<line_num-slines<<endl;
        }
        cout<<"Done! write to file: "<<outfilename<<endl;
        fin.close();fout.close();return 0;
    }

    void run()
    {
        int line_cnt = 1710672;
        int block_size = line_cnt/4;
        cout<<"block_size: "<<block_size<<endl;
        string outfilename[4]{"output data/Porto/grids1.txt", "output data/Porto/grids2.txt", "output data/Porto/grids3.txt", "output data/Porto/grids4.txt"};
        thread t1(Porto::splitLines, 0*block_size, 1*block_size, "data/Porto/train.csv", outfilename[0]);
        thread t2(Porto::splitLines, 1*block_size, 2*block_size, "data/Porto/train.csv", outfilename[1]);
        thread t3(Porto::splitLines, 2*block_size, 3*block_size, "data/Porto/train.csv", outfilename[2]);
        thread t4(Porto::splitLines, 3*block_size, line_cnt, "data/Porto/train.csv", outfilename[3]);
        t1.join();t2.join();t3.join();t4.join();
        // getchar();
    }

    void train_test_split(double p=0.8)
    {
        /*
            split the data to train data and test data according to the given probability.
        */
        ofstream ftrain(train_set);
        ofstream ftest(test_set);
        string outfilename[4]{"output data/Porto/grids1.txt", "output data/Porto/grids2.txt", "output data/Porto/grids3.txt", "output data/Porto/grids4.txt"};
        string line;
        double randomDouble;
        random_device rd;
        default_random_engine gen = default_random_engine(rd());
        uniform_real_distribution<double> dis(0,1);
        auto randfun = bind(dis, gen);
        cout<<randfun()<<"\t"<<randfun()<<endl;
        int allCnt=0;
        int testCnt=0;
        for (string &filename: outfilename) {
            ifstream fin(filename);
            while (getline(fin, line)) {
                randomDouble = randfun();
                ++allCnt;
                if (randomDouble>p) {
                    // test data
                    ftest<<line<<endl;
                    ++testCnt;
                } else {
                    // train data
                    ftrain<<line<<endl;
                }
            }
            fin.close();
            cout<<"process..."<<filename<<endl;
        }
        ftrain.close();
        ftest.close();
        cout<<"[Information] : all numbers: "<<allCnt<<"\ttest numbers: "<<testCnt<<"\tpercentage: "<<1.0*testCnt/allCnt<<endl;
    }


    int findSDPair(int start_gid, int end_gid, string outfilename)
    {
        /*
            Find the Source-Destination pair with exceed the given threshold.
            Here we extend the Source and Destination id, so that it could cover their neighbors and get more trajectories.
        */
        ofstream fout(outfilename.c_str());
        string infiles[]{"output data/Porto/grids1.txt", "output data/Porto/grids2.txt", "output data/Porto/grids3.txt", "output data/Porto/grids4.txt"};
        int cnt=0;
        string line;
        set<int> sid_set=getAreaNeighbors(start_gid);
        set<int> eid_set=getAreaNeighbors(end_gid);
        // for (int id:sid_set) cout<<" "<<id;cout<<endl;
        // for (int id:eid_set) cout<<" "<<id;cout<<endl;
        int s_gid, e_gid;
        int u_id;
        for (int i=0;i<4;++i)
        {
            ifstream fin(infiles[i].c_str());
            if (!fin) cout<<infiles[i]<<"not exist!"<<endl;
            int line_num=0;
            while (getline(fin, line))
            {
                int pos=0;
                DataUtils::getInt(line, pos, u_id);
                DataUtils::getInt(line, pos, s_gid);
                pos=line.rfind(" ");
                DataUtils::getInt(line, pos, e_gid);
                // cout<<"u_id: "<<u_id<<"\ts_gid: "<<s_gid<<"\te_gid: "<<e_gid<<endl;
                if (sid_set.find(s_gid)!=sid_set.end() && eid_set.find(e_gid)!=eid_set.end()) { fout<<line<<endl; ++cnt;}
                // if (++line_num>2) break;
            }
            fin.close();
        }
        fout.clear();
        cout<<"for start: "<<start_gid<<"\tend: "<<end_gid<<"\tcount: "<<cnt<<endl;
        return cnt;
    }

    void findSDThread()
    {
        thread t1(findSDPair, 152390, 137268, "output data/Porto/SDPair1.txt");  // 0.0359
        thread t2(findSDPair, 129685, 142254, "output data/Porto/SDPair2.txt");  // 0.0600
        thread t3(findSDPair, 142253, 129682, "output data/Porto/SDPair3.txt");  // 0.0549
        thread t4(findSDPair, 138096, 219516, "output data/Porto/SDPair4.txt");  // 0.12966
        thread t5(findSDPair, 134725, 153198, "output data/Porto/SDPair5.txt"); // 0.0619
        thread t6(findSDPair, 143110, 218677, "output data/Porto/SDPair6.txt");
        thread t7(findSDPair, 133890, 218677, "output data/Porto/SDPair7.txt");
        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
        t7.join();
    }

    void searchSDPair(string outfilename)
    {
        ofstream fout(outfilename.c_str());
        string infiles[]{"output data/Porto/grids1.txt", "output data/Porto/grids2.txt", "output data/Porto/grids3.txt", "output data/Porto/grids4.txt"};
        string line;
        for (int i=0;i<4;++i)
        {
            ifstream fin(infiles[i].c_str());
            if (!fin) cout<<infiles[i]<<"not exist!"<<endl;
            int line_num=0;
            while (getline(fin, line))
            {
                int pos=0;
                int start_id, end_id;
                pos=line.find(" ");
                DataUtils::getInt(line, pos, start_id);
                pos=line.rfind(" ");
                DataUtils::getInt(line, pos, end_id);
                int cnt=findSDPair(start_id, end_id, "output data/Porto/SDPair1.txt");
                if (cnt>100)
                {
                    fout<<start_id<<" "<<end_id<<" "<<cnt<<endl;
                    cout<<start_id<<" "<<end_id<<" "<<cnt<<endl;
                }
            }
            fin.close();
        }
        fout.close();
    }

    double calLCS(vector<int> &a, vector<int> &b)
    {
        /*
            LCSS: Longest common subsequence
            dp[i][j] = max({dp[i][j-1], dp[i-1][j], dp[i-1][j-1]+1 if a[i]==b[j]})
        */
        int ans=0;
        int n=a.size();
        int m=b.size();
        vector<vector<int>> dp(n+1, vector<int>(m+1,0));
        dp[0][0]=0;
        for (int i=0;i<n;++i)
        {
            for (int j=0;j<m;++j)
            {
                dp[i+1][j+1]=max({dp[i+1][j],dp[i][j+1], isConnect(a[i],b[j])?dp[i][j]+1:0});
            }
        }
        ans=dp[n][m];
        return 1.0*ans/(n+m-ans);
    }

    int calTrajSimilarityThread(vector<vector<int>> &trajs, vector<vector<double>> &sim, int k1, int k2, int n)
    {
        cout<<"calTrajSimilarity Thread from "<<k1<<" to "<<k2<<endl;
        for (int i=k1;i<k2;++i)
        {
            for (int j=i+1;j<n;++j)
            {
                sim[i][j]=sim[j][i]=1-calLCS(trajs[i], trajs[j]);
            }
        }
        return 0;
    }

    void calTrajSimilarity(string infilename, string outfilename)
    {
        ifstream fin(infilename.c_str());
        clock_t start_time = clock();
        if (!fin) {cout<<"Can not open \""<<infilename<<"\""<<endl; return;}
        vector<vector<int>> trajs;
        string line;
        while (getline(fin, line))
        {
            vector<int> tmp;
            int u_id;
            int pos=0;
            int id;
            DataUtils::getInt(line,pos,u_id);
            while (DataUtils::getInt(line, pos, id)) tmp.push_back(id);
            trajs.push_back(tmp);
        }
        fin.close();
        cout<<"load data from "<<infilename<<endl;
        cout<<"cost time: "<<(clock() - start_time)/CLOCKS_PER_SEC<<endl;
        start_time = clock();
        int n=trajs.size();
        vector<vector<double>> sim(n+1, vector<double>(n+1, 0));
        thread t1(calTrajSimilarityThread,ref(trajs), ref(sim), 0, n/16, n);
        thread t2(calTrajSimilarityThread, ref(trajs), ref(sim), n/16, n/16*3, n);
        thread t3(calTrajSimilarityThread, ref(trajs), ref(sim), n/16*3, n/16*5, n);
        thread t4(calTrajSimilarityThread, ref(trajs), ref(sim), n/16*5, n/16*8, n);
        thread t5(calTrajSimilarityThread, ref(trajs), ref(sim), n/16*8, n/16*12, n);
        thread t6(calTrajSimilarityThread, ref(trajs), ref(sim), n/16*12, n, n);
        t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();
        cout<<"cal Sim with size: "<<n<<" * "<<n<<endl;
        cout<<"cout time: "<<(clock()-start_time)/CLOCKS_PER_SEC<<endl;
        start_time=clock();
        ofstream fout(outfilename.c_str());
        for (int i=0;i<n;++i)
        {
            fout<<sim[i][0];for (int j=1;j<n;++j) fout<<" "<<sim[i][j];fout<<endl;
        }
        fout.close();
        cout<<"write similarity to "<<outfilename<<endl;
        cout<<"cost time "<<(clock()-start_time)/CLOCKS_PER_SEC<<endl;
    }

    void runCalTrajSimilarity()
    {
        calTrajSimilarity("output data/Porto/SDPair5.txt", "output data/Porto/SDSim5.txt");
        calTrajSimilarity("output data/Porto/SDPair4.txt", "output data/Porto/SDSim4.txt");
        calTrajSimilarity("output data/Porto/SDPair3.txt", "output data/Porto/SDSim3.txt");
        calTrajSimilarity("output data/Porto/SDPair2.txt", "output data/Porto/SDSim2.txt");
        calTrajSimilarity("output data/Porto/SDPair1.txt", "output data/Porto/SDSim1.txt");

        calTrajSimilarity("output data/Porto/SDPair6.txt", "output data/Porto/SDSim6.txt");
        calTrajSimilarity("output data/Porto/SDPair7.txt", "output data/Porto/SDSim7.txt");
    }

    void HierarchicalCluster(string infilename, string outfilename, double eta, double threshold)
    {
        clock_t start_time=clock();
        ifstream fin(infilename.c_str());
        if (!fin) {cout<<"Can not open "<<infilename<<endl; return;}
        vector<vector<double>> sim;
        string line;
        double dis;
        int pos;
        while (getline(fin, line))
        {
            pos=0;
            vector<double> tmp;
            while (DataUtils::getDouble(line, pos, dis)) tmp.push_back(dis);
            sim.push_back(tmp);
        }
        fin.close();
        int n=sim.size();
        cout<<"read similarity from "<<infilename<<" cost time: "<<(clock()-start_time)/CLOCKS_PER_SEC<<endl;
        start_time=clock();
        vector<int> ans = ML::HierarchicalCluster(sim, eta);
        cout<<"Hierarchical Cluster "<<infilename<<" size: "<<n<<"cost time: "<<(clock()-start_time)/CLOCKS_PER_SEC<<endl;
        start_time=clock();
        map<int, int> mp;
        for (int i:ans) ++mp[i];
        set<int> outliers;
        int outlier_cnt=0;
        for (auto i:mp)
        {
            if (1.0*i.second/n<threshold) outliers.insert(i.first), outlier_cnt+=i.second;
        }
        ofstream fout(outfilename.c_str());
        for (int i=0;i<n;++i)
        {
            if (outliers.find(ans[i])!=outliers.end()) fout<<i<<endl;
        }
        fout.close();
        cout<<"write outliers to "<<outfilename<<" cost time: "<<(clock()-start_time)/CLOCKS_PER_SEC<<endl;
        cout<<"Outliers percentage: "<<1.0*outlier_cnt/n<<endl;
    }

    void runHCThread()
    {
        thread t1(HierarchicalCluster, "output data/Porto/SDSim1.txt", "output data/Porto/SDOutliers1_HierarchicalCluster_eta_0.2_threshold_0.01.txt", 0.3, 0.003);
        thread t2(HierarchicalCluster, "output data/Porto/SDSim2.txt", "output data/Porto/SDOutliers2_HierarchicalCluster_eta_0.2_threshold_0.01.txt", 0.3, 0.003);
        thread t3(HierarchicalCluster, "output data/Porto/SDSim3.txt", "output data/Porto/SDOutliers3_HierarchicalCluster_eta_0.2_threshold_0.01.txt", 0.3, 0.003);
        thread t4(HierarchicalCluster, "output data/Porto/SDSim4.txt", "output data/Porto/SDOutliers4_HierarchicalCluster_eta_0.2_threshold_0.01.txt", 0.4, 0.003);
        thread t5(HierarchicalCluster, "output data/Porto/SDSim5.txt", "output data/Porto/SDOutliers5_HierarchicalCluster_eta_0.2_threshold_0.01.txt", 0.3, 0.006);  // eta=0.8, support=0.01 no outliers
        thread t6(HierarchicalCluster, "output data/Porto/SDSim6.txt", "output data/Porto/SDOutliers6_HierarchicalCluster_eta_0.2_threshold_0.01.txt", 0.4, 0.003);
        thread t7(HierarchicalCluster, "output data/Porto/SDSim7.txt", "output data/Porto/SDOutliers7_HierarchicalCluster_eta_0.2_threshold_0.01.txt", 0.6, 0.003);
        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
        t7.join();
    }

    void testOutliersPercent(string infilename, string outlierfilename)
    {
        ifstream fin(infilename.c_str());
        int n=0;
        string line;
        while (getline(fin, line)) ++n;
        fin.close();
        ifstream fin2(outlierfilename.c_str());
        int n1=0;
        while (getline(fin2, line)) ++n1;
        fin2.close();
        cout<<"Outliers Percentage: "<<outlierfilename<<"  "<<1.0*n1/n<<endl;
    }

    void testALL()
    {
        testOutliersPercent("output data/Porto/SDSim1.txt", "output data/Porto/SDOutliers1_HierarchicalCluster_eta_0.2_threshold_0.01.txt");
        testOutliersPercent("output data/Porto/SDSim2.txt", "output data/Porto/SDOutliers2_HierarchicalCluster_eta_0.2_threshold_0.01.txt");
        testOutliersPercent("output data/Porto/SDSim3.txt", "output data/Porto/SDOutliers3_HierarchicalCluster_eta_0.2_threshold_0.01.txt");
        testOutliersPercent("output data/Porto/SDSim4.txt", "output data/Porto/SDOutliers4_HierarchicalCluster_eta_0.2_threshold_0.01.txt");
        testOutliersPercent("output data/Porto/SDSim5.txt", "output data/Porto/SDOutliers5_HierarchicalCluster_eta_0.2_threshold_0.01.txt");
        testOutliersPercent("output data/Porto/SDSim6.txt", "output data/Porto/SDOutliers6_HierarchicalCluster_eta_0.2_threshold_0.01.txt");
        testOutliersPercent("output data/Porto/SDSim7.txt", "output data/Porto/SDOutliers7_HierarchicalCluster_eta_0.2_threshold_0.01.txt");
    }


    void Main()
    {
        run();
        findSDThread();
        runCalTrajSimilarity();
        runHCThread();
        testALL();
    }
};

int main()
{
    cout<<__cplusplus<<endl;

    Porto::Main();

    getchar();
    return 0;
}
