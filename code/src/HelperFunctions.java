import java.util.Arrays;

public class HelperFunctions {
	
	
    /**
     * Calculates the positive remainder of a divided by b.
     * @param a
     * @param b
     * @return
     */
    public static int posMod(int a, int b) {

        int result = Math.floorMod(a, b);

        if (result < 0)
                result += b;

        return result;


    }
	
    /**
     * Calculates ln(a+b), given ln(a) and ln(b)
     * @param lna
     * @param lnb
     * @return
     */
    public static double logSum(double lna, double lnb) {
    	
    	if (lna >= lnb)
    		return lna + Math.log(1 + Math.exp(lnb - lna));
    	else
    		return lnb + Math.log(1 + Math.exp(lna - lnb));

    	
    }
    
    
    /**
     * Calculates the median of unsorted array a
     * @param a
     * @return
     */
    public static double getMedian(double[] a) {
    	
    	Arrays.sort(a);
    	int n = a.length;
    	
    	if (n % 2 == 0) {
    		double m1 = a[n/2];
    		double m2 = a[n/2+1];
    		return (m1 + m2) / 2;
    	} else {
    		return a[(n+1)/2];
    	}
    		
    	
    		
    	
    }

}