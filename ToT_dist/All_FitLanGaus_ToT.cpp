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


vector<double> load_txt_file(const string& directory, int col, int rows_to_draw); 
double langaufun(double *x, double *par);
TF1 *langaufit(TH1F *his, double *fitrange, double *startvalues, double *parlimitslo, double *parlimitshi, double *fitparams, double *fiterrors, double *ChiSqr, int *NDF);
int langaupro(double *params, double &maxx, double &FWHM);




void plot_tot_histogram(std::string directory = "/Users/yoonha/cernbox/A-STEP/241005/ToT_distributions_THR200_APS3-W08-S03_20241005_175622") {

    std::vector<int> cols_to_draw;  // Default columns
    std::vector<int> rows_to_draw;  // Default rows_to_draw

    cols_to_draw = {32, 33, 34};
    rows_to_draw = {32, 33, 34};

    const int nc = cols_to_draw.size();  // Number of columns
    const int nr = rows_to_draw.size();  // Number of rows_to_draw

    // Create canvas and divide it into nc*nr pads
    TCanvas* c1 = new TCanvas("c1", "ToT Dist.", 200 * nc, 200 * nr);
    c1->Divide(nc, nr, 0, 0);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(111);
    gErrorIgnoreLevel = kWarning;

    // Create a histogram to hold all the values
    TH1F* hist_tot_all = new TH1F("hist_tot_all", "Total Histogram", 20, 0, 20);
    TH1F* hist_tot_pixel[35][35];
    TF1* fit_tot_pixel[35][35];

    TH1F* hist_MPV = new TH1F("hist_MPV", "MPV Distribution", 50, 0, 10);
    float MPV[35][35];

    double fr[2];
    fr[0] = 0;//fr[0] = 0.3 * hist_tot_all->GetMean();
    fr[0] = 20;//fr[1] = 3.0 * hist_tot_all->GetMean();

    // Width, MP, Area, GSigma
    double pllo[4] = {0.09, 0.01, 5, 0.2};
    double plhi[4] = {2.80, 20.0, 10000, 2.8};
    double sv[4] = {0.5, 4.0, 500, 0.4};

    double fp[4], fpe[4];
    double chisqr;
    int ndf;
    double SNRPeak, SNRFWHM;
    

    for (int r=0; r<35; r++) {
        for (int c=0; c<35; c++) {
            // Load data for the current column and rows_to_draw
            std::vector<double> data = load_txt_file(directory, c, r);

            hist_tot_pixel[c][r] = new TH1F(Form("c%d_r%d", c, r), Form("c%d_r%d", c, r), 20, 0, 20);

            if (data.empty()){
                continue;
            }
            for (double value : data)
            {
                hist_tot_pixel[c][r]->Fill(value);
                hist_tot_all->Fill(value); // Fill the total histogram
            }

            fit_tot_pixel[c][r] = langaufit(hist_tot_pixel[c][r], fr, sv, pllo, plhi, fp, fpe, &chisqr, &ndf);
            langaupro(fp, SNRPeak, SNRFWHM);
            MPV[c][r] = fp[1];

            if(MPV[c][r]<1e-3) std::cout << c << ' ' << r << "  Small MPV! " <<std::endl ;

            hist_MPV -> Fill( fp[1] );


        }
    }


    int index = 0;
    for (int row: rows_to_draw) {
        for (int col : cols_to_draw) {
            printf("%f, ", MPV[col][row]);
            c1 -> cd(index+1);
            hist_tot_pixel[col][row]->Draw();
            if(MPV[col][row] > pllo[1] ) fit_tot_pixel[col][row]->Draw("lsame");
            index++;
        }
        printf("\n");
    }

    TCanvas* c2 = new TCanvas("c2", "MPV Dist.", 800, 800);
    c2 -> cd();
    hist_MPV -> Draw();

    TFitResultPtr fit_result = hist_MPV->Fit("gaus", "SQ");

    std::string outdir;
    std::string figdir = outdir.empty() ? "." : outdir;
    std::string file_name = gSystem->BaseName(directory.c_str());
    c1->cd(0);
    c1->SaveAs((figdir + "/" + file_name + ".pdf").c_str());

}

// Function to load a .txt file and return a vector of values
std::vector<double> load_txt_file(const std::string& directory, int col, int row) {
    std::string filename = "ToT_distribution_col" + std::to_string(col) + "_row" + std::to_string(row) + ".txt";
    std::string filepath = directory + "/" + filename;

    std::ifstream file(filepath);
    std::vector<double> data;

    if (!file.is_open()) {
        //std::cerr << "File " << filepath << " not found." << std::endl;
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
