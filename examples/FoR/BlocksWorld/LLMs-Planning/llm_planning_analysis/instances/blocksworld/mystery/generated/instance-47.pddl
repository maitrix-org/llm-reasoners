(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a j i e)
(:init 
(harmony)
(planet a)
(planet j)
(planet i)
(planet e)
(province a)
(province j)
(province i)
(province e)
)
(:goal
(and
(craves a j)
(craves j i)
(craves i e)
)))