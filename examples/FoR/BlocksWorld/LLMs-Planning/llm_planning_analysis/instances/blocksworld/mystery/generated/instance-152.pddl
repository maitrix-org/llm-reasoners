(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b i f e)
(:init 
(harmony)
(planet b)
(planet i)
(planet f)
(planet e)
(province b)
(province i)
(province f)
(province e)
)
(:goal
(and
(craves b i)
(craves i f)
(craves f e)
)))