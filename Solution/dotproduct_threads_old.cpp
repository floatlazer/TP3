# include <vector>
# include <cassert>
# include <string>
# include <iostream>
# include <chrono>
# include <thread>

double dot( std::vector<double>& u, std::vector<double>& v )
{
  assert(u.size() == v.size());
  double scal = 0.;
  unsigned num_cpus = std::thread::hardware_concurrency();
  int sz = u.size()/num_cpus;
  int reste = u.size()%num_cpus;
  if ( reste > 0 ) sz += 1;
  int start = 0;
  std::vector<std::thread> threads;
  threads.reserve(num_cpus);
  std::vector<double> partial_sums(num_cpus, 0.);
  for ( unsigned p = 0; p < num_cpus-1; ++p ) {
      int size = ( p >= reste ? sz : sz-1 );
      threads.push_back(std::thread( [&partial_sums,&u,&v,p] (int start, int sz) -> void {
	  for( int i = start; i < start+sz; ++i )
	    partial_sums[p] += u[i]*v[i];
	  }, start, size));
      start += size;
  }
  for (int i = start; i < u.size(); ++i ) partial_sums[num_cpus-1] += u[i]*v[i];
  for ( auto& p : threads ) p.join();
  for ( auto& ps : partial_sums ) {
    scal += ps;
  }
  return scal;
}

int main( int nargs, char* vargs[])
{
  std::chrono::time_point<std::chrono::system_clock> start, end;
  int N = 1023;
  int nbSamples = 1024;
  if (nargs > 1) {
    nbSamples = std::stoi(vargs[1]);
  }
  
  start = std::chrono::system_clock::now();
  std::vector<std::vector<double>> U(nbSamples), V(nbSamples);
  for ( int iSample = 0; iSample < nbSamples; ++iSample ) {
    U[iSample] = std::vector<double>(N);
    V[iSample] = std::vector<double>(N);
    for ( int i = 0; i < N; ++i ) {
      U[iSample][i] = (iSample + 1 + i)%N;
      V[iSample][i] = (nbSamples - iSample + i)%N;
    }
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Temps assemblage vecteurs : " << elapsed_seconds.count() 
              << std::endl;

  start = std::chrono::system_clock::now();
  std::vector<double> result(nbSamples);
  for ( int iSample = 0; iSample < nbSamples; ++iSample )
    result[iSample] = dot(U[iSample],V[iSample]);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  std::cout << "Temps produits scalaires : " << elapsed_seconds.count() 
              << std::endl;

  start = std::chrono::system_clock::now();
  double ref = result[0];
  double sum = 0;;
  for ( const auto& val : result )
    sum += val;
  sum /= ref;
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  std::cout << "Temps sommation : " << elapsed_seconds.count() 
              << std::endl;
  std::cout << "sum : " << sum << std::endl;
  return 0;
}
