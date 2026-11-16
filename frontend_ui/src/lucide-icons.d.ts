declare module 'lucide-react/dist/esm/icons/*.mjs' {
  import type { LucideProps } from 'lucide-react';
  import type { ForwardRefExoticComponent, RefAttributes } from 'react';

  const Icon: ForwardRefExoticComponent<LucideProps & RefAttributes<SVGSVGElement>>;
  export default Icon;
}
