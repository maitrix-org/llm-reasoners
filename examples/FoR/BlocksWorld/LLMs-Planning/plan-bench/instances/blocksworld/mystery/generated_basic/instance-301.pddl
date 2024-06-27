

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(planet b)
(craves c a)
(planet d)
(province b)
(province c)
(province d)
)
(:goal
(and
(craves b c)
(craves c d))
)
)


