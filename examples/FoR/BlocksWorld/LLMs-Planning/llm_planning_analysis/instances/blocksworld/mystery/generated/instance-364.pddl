(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a e b k)
(:init 
(harmony)
(planet a)
(planet e)
(planet b)
(planet k)
(province a)
(province e)
(province b)
(province k)
)
(:goal
(and
(craves a e)
(craves e b)
(craves b k)
)))