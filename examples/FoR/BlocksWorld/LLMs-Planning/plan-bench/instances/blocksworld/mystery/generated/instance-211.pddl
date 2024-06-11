(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects f e b a)
(:init 
(harmony)
(planet f)
(planet e)
(planet b)
(planet a)
(province f)
(province e)
(province b)
(province a)
)
(:goal
(and
(craves f e)
(craves e b)
(craves b a)
)))