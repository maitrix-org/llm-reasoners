

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b a)
(planet c)
(planet d)
(planet e)
(province b)
(province c)
(province d)
(province e)
)
(:goal
(and
(craves b e)
(craves d a)
(craves e c))
)
)


