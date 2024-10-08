#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <TCanvas.h>
#include <TH1F.h>
#include <TFile.h>
#include <TString.h>
#include <TSystem.h>
#include <TROOT.h>

using namespace std;

vector<double> load_txt_file(const string& directory, int col, int row); 
double langaufun(double *x, double *par);

void Single_Fit_ToT(string directory = "/Users/yoonha/cernbox/A-STEP/241005/ToT_distributions_THR200_APS3-W08-S03_20241005_175622") {

    vector<int> cols; 
    vector<int> rows;   

    for(int i=0; i<35; i++){
        cols.push_back(i);
        rows.push_back(i);
    }

    //cols = {15, 16, 17, 18};  
    //rows = {15, 16, 17, 18};
    int nc = cols.size();  // Number of columns
    int nr = rows.size();  // Number of rows

    // Create canvas and divide it into nc*nr pads
    TCanvas* c1 = new TCanvas("c1", "Dynamic Filling Example", 800, 800);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(111);
    gErrorIgnoreLevel = kWarning;
    // Create a histogram to hold all the values
    TH1F* hist_tot_all = new TH1F("hist_tot_all", "Total Histogram", 105, 0, 21);
    hist_tot_all -> Sumw2();
    vector<TH1F*> hist_tot_pixel(nc * nr, nullptr);

    int index = 0;
    for (int row : rows) {
        for (int col : cols) {
            // Load data for the current column and row
            vector<double> data = load_txt_file(directory, col, row);

            hist_tot_pixel[index] = new TH1F(Form("c%d_r%d", col, row), Form("c%d_r%d", col, row), 20, 0, 20);
            hist_tot_pixel[index]->Sumw2();

            if (!data.empty()) {
                for (double value : data) {
                    hist_tot_pixel[index]->Fill(value);
                    hist_tot_all->Fill(value);  // Fill the total histogram
                }
            }

            hist_tot_pixel[index]->SetMarkerStyle(4);
            hist_tot_pixel[index]->Draw("E");
            index++;
        }
    }
    double pllo[4] = {0.09, 2.0, 50, 0.2};
    double sv[4] = {0.5, 4.0, 500, 0.4};
    double plhi[4] = {0.80, 20.0, 10000, 4.8};

    TF1 *fitFunc = new TF1("fitFunc", langaufun, 2, 10, 4);
    fitFunc -> SetParNames("Width","MP","Area","GSigma");
    fitFunc -> SetParameters(sv);
    //for(int par=0; par<4; par++) fitFunc -> SetParLimits( par, pllo[par], plhi[par]);
    hist_tot_all->Fit("fitFunc", "RQ");

    TF1 *landaur = new TF1("landaur", "landau", 2, 20);
    //hist_tot_all->Fit("landaur", "r");

    double fitparams[4], fiterrors[4];
    double ChiSqr, NDF;
    fitFunc->GetParameters(fitparams); // obtain fit parameters
    for (int i = 0; i < 4; i++) fiterrors[i] = fitFunc->GetParError(i); // obtain fit parameter errors
    ChiSqr = fitFunc->GetChisquare(); // obtain chi^2
    NDF = fitFunc->GetNDF();          // obtain ndf

    // Global style settings
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(111);
    gStyle->SetLabelSize(0.03, "x");
    gStyle->SetLabelSize(0.03, "y");

    hist_tot_all->Draw("E");
}

// Function to load a .txt file and return a vector of values
vector<double> load_txt_file(const string& directory, int col, int row) {
    string filename = "ToT_distribution_col" + to_string(col) + "_row" + to_string(row) + ".txt";
    string filepath = directory + "/" + filename;

    ifstream file(filepath);
    vector<double> data;

    if (!file.is_open()) {
        cerr << "File " << filepath << " not found." << endl;
        return data;  // Return empty vector if file not found
    }

    double value;
    while (file >> value) {
        data.push_back(value);  // Read values from the file
    }

    return data;
}

double langaufun(double *x, double *par) {
 
   //Fit parameters:
   //par[0]=Width (scale) parameter of Landau density
   //par[1]=Most Probable (MP, location) parameter of Landau density
   //par[2]=Total area (integral -inf to inf, normalization constant)
   //par[3]=Width (sigma) of convoluted Gaussian function
   //
   //In the Landau distribution (represented by the CERNLIB approximation),
   //the maximum is located at x=-0.22278298 with the location parameter=0.
   //This shift is corrected within this function, so that the actual
   //maximum is identical to the MP parameter.
 
      // Numeric constants
      double invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
      double mpshift  = -0.22278298;       // Landau maximum location
 
      // Control constants
      double np = 100.0;      // number of convolution steps
      double sc =   5.0;      // convolution extends to +-sc Gaussian sigmas
 
      // Variables
      double xx;
      double mpc;
      double fland;
      double sum = 0.0;
      double xlow,xupp;
      double step;
      double i;
 
 
      // MP shift correction
      mpc = par[1] - mpshift * par[0];
 
      // Range of convolution integral
      xlow = x[0] - sc * par[3];
      xupp = x[0] + sc * par[3];
 
      step = (xupp-xlow) / np;
 
      // Convolution integral of Landau and Gaussian by sum
      for(i=1.0; i<=np/2; i++) {
         xx = xlow + (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
 
         xx = xupp - (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
      }
 
      return (par[2] * step * sum * invsq2pi / par[3]);
}
 

