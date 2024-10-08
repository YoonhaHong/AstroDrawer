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
TF1 *langaufit(TH1F *his, double *fitrange, double *startvalues, double *parlimitslo, double *parlimitshi, double *fitparams, double *fiterrors, double *ChiSqr, int *NDF);
int langaupro(double *params, double &maxx, double &FWHM);


void FitLanGaus_ToT(string directory = "/Users/yoonha/cernbox/A-STEP/241005/ToT_distributions_THR200_APS3-W08-S03_20241005_175622") {

    vector<int> cols; 
    vector<int> rows;   

    for(int i=20; i<35; i++){
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

    // Setting fit range and start values
    double fr[2];
    fr[0] = 0;//fr[0] = 0.3 * hist_tot_all->GetMean();
    fr[0] = 20;//fr[1] = 3.0 * hist_tot_all->GetMean();

    // Width, MP, Area, GSigma
    double pllo[4] = {0.09, 2.0, 50, 0.2};
    double plhi[4] = {0.80, 20.0, 10000, 4.8};
    double sv[4] = {0.5, 4.0, 500, 0.4};

    double fp[4], fpe[4];
    double chisqr;
    int ndf;
    TF1 *fitsnr = langaufit(hist_tot_all, fr, sv, pllo, plhi, fp, fpe, &chisqr, &ndf);

    double SNRPeak, SNRFWHM;
    langaupro(fp, SNRPeak, SNRFWHM);

    // Global style settings
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(111);
    gStyle->SetLabelSize(0.03, "x");
    gStyle->SetLabelSize(0.03, "y");

    //hist_tot_all->GetXaxis()->SetRange(0, 70);
    hist_tot_all->Draw("E");
    fitsnr->Draw("lsame");
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
 
 
 
TF1 *langaufit(TH1F *his, double *fitrange, double *startvalues, double *parlimitslo, double *parlimitshi, double *fitparams, double *fiterrors, double *ChiSqr, int *NDF)
{
   // Once again, here are the Landau * Gaussian parameters:
   //   par[0]=Width (scale) parameter of Landau density
   //   par[1]=Most Probable (MP, location) parameter of Landau density
   //   par[2]=Total area (integral -inf to inf, normalization constant)
   //   par[3]=Width (sigma) of convoluted Gaussian function
   //
   // Variables for langaufit call:
   //   his             histogram to fit
   //   fitrange[2]     lo and hi boundaries of fit range
   //   startvalues[4]  reasonable start values for the fit
   //   parlimitslo[4]  lower parameter limits
   //   parlimitshi[4]  upper parameter limits
   //   fitparams[4]    returns the final fit parameters
   //   fiterrors[4]    returns the final fit errors
   //   ChiSqr          returns the chi square
   //   NDF             returns ndf
 
   int i;
   char FunName[100];
 
   sprintf(FunName,"Fitfcn_%s",his->GetName());
 
   TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
   if (ffitold) delete ffitold;
 
   TF1 *ffit = new TF1(FunName,langaufun,fitrange[0],fitrange[1],4);
   ffit->SetParameters(startvalues);
   ffit->SetParNames("Width","MP","Area","GSigma");
 
   for (i=0; i<4; i++) {
      ffit->SetParLimits(i, parlimitslo[i], parlimitshi[i]);
   }
 
   his->Fit(FunName,"RB0Q");   // fit within specified range, use ParLimits, do not plot
 
   ffit->GetParameters(fitparams);    // obtain fit parameters
   for (i=0; i<4; i++) {
      fiterrors[i] = ffit->GetParError(i);     // obtain fit parameter errors
   }
   ChiSqr[0] = ffit->GetChisquare();  // obtain chi^2
   NDF[0] = ffit->GetNDF();           // obtain ndf
 
   return (ffit);              // return fit function
 
}
 
 
int langaupro(double *params, double &maxx, double &FWHM) {
 
   // Searches for the location (x value) at the maximum of the
   // Landau-Gaussian convolute and its full width at half-maximum.
   //
   // The search is probably not very efficient, but it's a first try.
 
   double p,x,fy,fxr,fxl;
   double step;
   double l,lold;
   int i = 0;
   int MAXCALLS = 10000;
 
 
   // Search for maximum
 
   p = params[1] - 0.1 * params[0];
   step = 0.05 * params[0];
   lold = -2.0;
   l    = -1.0;
 
 
   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;
 
      lold = l;
      x = p + step;
      l = langaufun(&x,params);
 
      if (l < lold)
         step = -step/10;
 
      p += step;
   }
 
   if (i == MAXCALLS)
      return (-1);
 
   maxx = x;
 
   fy = l/2;
 
 
   // Search for right x location of fy
 
   p = maxx + params[0];
   step = params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;
 
 
   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;
 
      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }
 
   if (i == MAXCALLS)
      return (-2);
 
   fxr = x;
 
 
   // Search for left x location of fy
 
   p = maxx - 0.5 * params[0];
   step = -params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;
 
   while ( (l != lold) && (i < MAXCALLS) ) {
      i++;
 
      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }
 
   if (i == MAXCALLS)
      return (-3);
 
 
   fxl = x;
 
   FWHM = fxr - fxl;
   return (0);
}