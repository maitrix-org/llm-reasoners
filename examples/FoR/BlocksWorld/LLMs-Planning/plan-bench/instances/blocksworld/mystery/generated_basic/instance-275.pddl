

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b d)
(planet c)
(planet d)
(province a)
(province b)
(province c)
)
(:goal
(and
(craves a b)
(craves b c)
(craves c d))
)
)


