

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b a)
(craves c d)
(planet d)
(province b)
(province c)
)
(:goal
(and
(craves b a)
(craves d c))
)
)


