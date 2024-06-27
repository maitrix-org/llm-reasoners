

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b c)
(on c d)
(on d a)
(clear b)
)
(:goal
(and
(on a c)
(on d a))
)
)


